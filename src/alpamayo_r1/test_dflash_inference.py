#!/usr/bin/env python3
"""Test script for DFlash-accelerated Alpamayo inference.

This script compares standard autoregressive inference with DFlash speculative decoding
for Chain-of-Causation (CoC) generation.

Usage:
    python test_dflash_inference.py [--compare] [--verbose] [--draft-model PATH]

Options:
    --compare       Run both standard and DFlash inference for comparison
    --verbose       Run in verbose debug mode - shows step-by-step token details
                    Output format per step:
                      Step N: Draft tokens [id1, id2, ...] = ['tok1', 'tok2', ...]
                              Target tokens [id1, id2, ...] = ['tok1', 'tok2', ...]
                              Accepted: M tokens
    --draft-model   Path to DFlash draft model (default: /models/Qwen3-8B-DFlash-b16)
    --output-dir    Directory to save experiment results (default: /exp)

Example:
    # Run verbose debug on single clip
    python test_dflash_inference.py --verbose --clip-id <clip-id> --max-tokens 50
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.dflash_integration import (
    DFlashAlpamayoAccelerator,
    DFlashConfig,
    load_dflash_draft_model,
)
from alpamayo_r1.models.token_utils import to_special_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_standard_inference(
    model: AlpamayoR1,
    model_inputs: dict,
    temperature: float = 0.6,
    max_tokens: int = 256,
) -> tuple[str, float, torch.Tensor, torch.Tensor]:
    """Run standard autoregressive inference.

    Returns:
        Tuple of (coc_text, time_ms, pred_xyz, pred_rot).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=temperature,
            num_traj_samples=1,
            max_generation_length=max_tokens,
            return_extra=True,
        )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    coc_text = extra["cot"][0, 0, 0] if extra["cot"].ndim == 3 else extra["cot"][0][0]

    return coc_text, elapsed_ms, pred_xyz, pred_rot


def run_dflash_inference(
    model: AlpamayoR1,
    accelerator: DFlashAlpamayoAccelerator,
    tokenized_data: dict,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Run DFlash-accelerated inference for CoC generation only.

    Note: This only generates the CoC text, not the full trajectory.
    The trajectory generation would use the standard diffusion sampling.

    Returns:
        Tuple of (coc_text, time_ms, stats_dict).
    """
    # Stop at <|cot_end|> - draft model wasn't trained on tokens after this
    # This avoids wasting iterations on <|cot_end|> -> <|traj_future_start|> -> trajectory
    stop_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("cot_end"))

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids, stats = accelerator.generate(
            input_ids=tokenized_data["input_ids"],
            pixel_values=tokenized_data.get("pixel_values"),
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            max_new_tokens=max_tokens,
            stop_token_ids=[stop_token_id],
            temperature=temperature,
        )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Decode the generated text
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract CoC portion (between <|cot_start|> and <|cot_end|> or <|traj_future_start|>)
    coc_start = "<|cot_start|>"
    coc_end = "<|cot_end|>"
    traj_start = "<|traj_future_start|>"

    coc_text = generated_text
    if coc_start in generated_text:
        coc_text = generated_text.split(coc_start)[-1]
    if coc_end in coc_text:
        coc_text = coc_text.split(coc_end)[0]
    elif traj_start in coc_text:
        coc_text = coc_text.split(traj_start)[0]

    coc_text = coc_text.strip()

    stats_dict = {
        "total_tokens": stats.total_tokens,
        "total_iterations": stats.total_iterations,
        "mean_acceptance_length": stats.mean_acceptance_length,
        "acceptance_rate": stats.acceptance_rate,
        "prefill_time_ms": stats.prefill_time_ms,
        "decode_time_ms": stats.decode_time_ms,
    }

    return coc_text, elapsed_ms, stats_dict


@torch.inference_mode()
def run_dflash_inference_verbose(
    model: AlpamayoR1,
    accelerator: DFlashAlpamayoAccelerator,
    tokenized_data: dict,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict:
    """Run DFlash inference with detailed step-by-step debug output.

    This function shows exactly what happens at each decode step:
    - Draft tokens predicted by the draft model
    - Target tokens sampled from the target model
    - Which tokens were accepted and why

    Returns:
        Dict with detailed step information, final results, and log string.
    """
    from alpamayo_r1.dflash_integration import (
        extract_context_feature,
        sample_tokens,
    )
    from transformers import DynamicCache

    # Build log as list of lines
    log_lines = []
    def log(msg=""):
        log_lines.append(msg)
        print(msg)

    device = tokenized_data["input_ids"].device
    block_size = accelerator.block_size
    mask_token_id = accelerator.mask_token_id
    target_layer_ids = accelerator.target_layer_ids

    # Stop at <|cot_end|> - draft model wasn't trained on tokens after this
    stop_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("cot_end"))

    # Initialize
    input_ids = tokenized_data["input_ids"]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )

    past_key_values_target = DynamicCache()

    # Track steps
    steps = []

    log()
    log("=" * 80)
    log("DFLASH INFERENCE - VERBOSE DEBUG MODE")
    log("=" * 80)
    log(f"Block size: {block_size}")
    log(f"Mask token ID: {mask_token_id}")
    log(f"Max new tokens: {max_tokens}")
    log()

    # ====== PREFILL ======
    log(">>> PREFILL STAGE")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefill_output = accelerator.target_vlm(
            input_ids=input_ids,
            pixel_values=tokenized_data.get("pixel_values"),
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

    rope_deltas = getattr(accelerator.target_vlm.model, "rope_deltas", None)
    output_ids[:, :num_input_tokens] = input_ids

    # Sample first token
    first_token_logits = prefill_output.logits[:, -1:, :]
    first_token = sample_tokens(first_token_logits, temperature, accelerator._logits_processor)
    output_ids[:, num_input_tokens] = first_token[0, 0]

    first_token_id = first_token[0, 0].item()
    first_token_str = model.tokenizer.decode([first_token_id])
    log(f"First token: {first_token_id} = '{first_token_str}'")

    # Extract context
    full_hidden = extract_context_feature(prefill_output.hidden_states, target_layer_ids)
    target_hidden = full_hidden[:, -1:, :]

    log(f"Input length: {num_input_tokens}")
    log()

    # ====== DECODE LOOP ======
    start_pos = num_input_tokens
    step_num = 0

    while start_pos < max_length:
        step_num += 1
        log("-" * 80)
        log(f">>> STEP {step_num} (position {start_pos})")

        # Get current block
        block_output_ids = output_ids[:, start_pos : start_pos + block_size].clone()
        first_in_block = block_output_ids[0, 0].item()
        first_in_block_str = model.tokenizer.decode([first_in_block])

        log(f"  First token in block: {first_in_block} = '{first_in_block_str}'")

        # Position IDs for target model
        block_positions = torch.arange(start_pos, start_pos + block_size, device=device)
        if rope_deltas is not None:
            block_position_ids = block_positions.view(1, 1, -1).expand(3, 1, -1).clone()
            block_position_ids = block_position_ids + rope_deltas.unsqueeze(-1).to(dtype=torch.long, device=device)
        else:
            block_position_ids = block_positions.unsqueeze(0)

        # Position IDs for draft model (reset)
        reset_position_ids = torch.arange(0, 1 + block_size, device=device).unsqueeze(0)

        # Draft model forward
        noise_embedding = accelerator._embed_tokens(block_output_ids)
        current_context = target_hidden[:, -1:, :]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            draft_hidden = accelerator.draft_model(
                target_hidden=current_context,
                noise_embedding=noise_embedding,
                position_ids=reset_position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )

        # Get draft predictions
        draft_logits = accelerator._lm_head(draft_hidden[:, 1:, :])
        draft_tokens = sample_tokens(draft_logits, temperature, accelerator._logits_processor)
        block_output_ids[:, 1:] = draft_tokens

        # Decode draft tokens for display
        draft_token_ids = block_output_ids[0].tolist()
        draft_token_strs = [model.tokenizer.decode([t]) for t in draft_token_ids]

        log(f"  Draft tokens:  {draft_token_ids}")
        log(f"  Draft decoded: {draft_token_strs}")

        # Target model verification
        with torch.autocast("cuda", dtype=torch.bfloat16):
            verify_output = accelerator.target_vlm(
                input_ids=block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

        posterior = sample_tokens(verify_output.logits, temperature, accelerator._logits_processor)
        posterior_ids = posterior[0].tolist()
        posterior_strs = [model.tokenizer.decode([t]) for t in posterior_ids]

        log(f"  Target tokens: {posterior_ids}")
        log(f"  Target decoded:{posterior_strs}")

        # Compute acceptance
        matches = block_output_ids[:, 1:] == posterior[:, :-1]
        match_list = matches[0].tolist()
        acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

        log(f"  Matches:       {match_list}")
        log(f"  Accepted: {acceptance_length + 1} tokens (draft={acceptance_length}, +1 from target)")

        # Accept tokens
        output_ids[:, start_pos : start_pos + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start_pos + acceptance_length + 1] = posterior[:, acceptance_length]

        # Show accepted tokens
        accepted_ids = output_ids[0, start_pos : start_pos + acceptance_length + 2].tolist()
        accepted_strs = [model.tokenizer.decode([t]) for t in accepted_ids]
        log(f"  Final accepted IDs:  {accepted_ids}")
        log(f"  Final accepted text: {accepted_strs}")

        # Check stop condition BEFORE recording step
        # The last accepted token is posterior[0, acceptance_length]
        last_accepted_token = posterior[0, acceptance_length].item()
        hit_stop_token = (stop_token_id == last_accepted_token)

        # Record step info
        step_info = {
            "step": step_num,
            "position": start_pos,
            "first_token": {"id": first_in_block, "text": first_in_block_str},
            "draft_tokens": {"ids": draft_token_ids, "text": draft_token_strs},
            "target_tokens": {"ids": posterior_ids, "text": posterior_strs},
            "matches": match_list,
            "acceptance_length": acceptance_length + 1,
            "accepted_tokens": {"ids": accepted_ids, "text": accepted_strs},
        }
        steps.append(step_info)

        # Stop immediately if we hit the stop token
        if hit_stop_token:
            log(f"\n  >>> STOP TOKEN (<|cot_end|>) accepted, stopping")
            start_pos += acceptance_length + 1
            break

        # Update position
        start_pos += acceptance_length + 1

        # Update cache
        past_key_values_target.crop(start_pos)

        # Update target hidden
        new_hidden = extract_context_feature(verify_output.hidden_states, target_layer_ids)
        target_hidden = new_hidden[:, acceptance_length : acceptance_length + 1, :]

        if step_num >= 50:  # Safety limit for verbose mode
            log(f"\n  >>> REACHED MAX STEPS (50)")
            break

    log()
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)

    # Final output
    output_ids = output_ids[:, :max_length]
    mask = output_ids[0] != mask_token_id
    output_ids = output_ids[:, mask]

    # Truncate at stop token
    generated_tokens = output_ids[0, num_input_tokens:]
    if stop_token_id in generated_tokens:
        stop_idx = (generated_tokens == stop_token_id).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, :num_input_tokens + stop_idx[0] + 1]

    total_tokens = output_ids.shape[1] - num_input_tokens
    total_iterations = len(steps)
    acceptance_lengths = [s["acceptance_length"] for s in steps]
    mean_acceptance = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0
    total_drafted = total_iterations * (block_size - 1)
    total_accepted = sum(max(0, a - 1) for a in acceptance_lengths)
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    log(f"Total tokens generated: {total_tokens}")
    log(f"Total iterations: {total_iterations}")
    log(f"Mean acceptance length: {mean_acceptance:.2f}")
    log(f"Acceptance rate: {acceptance_rate:.1%}")
    log()

    # Per-step summary
    log("Per-step acceptance:")
    for s in steps:
        log(f"  Step {s['step']:2d}: accepted {s['acceptance_length']} tokens - "
            f"first='{s['first_token']['text']}' ({s['first_token']['id']})")

    # Decode final output
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    log(f"\nGenerated text:\n{generated_text}")

    return {
        "steps": steps,
        "total_tokens": total_tokens,
        "total_iterations": total_iterations,
        "mean_acceptance_length": mean_acceptance,
        "acceptance_rate": acceptance_rate,
        "output_ids": output_ids,
        "generated_text": generated_text,
        "log": "\n".join(log_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Test DFlash-accelerated Alpamayo inference")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both standard and DFlash inference for comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run in verbose debug mode - shows step-by-step token details",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/models/Qwen3-8B-DFlash-b16",
        help="Path to DFlash draft model (local path)",
    )
    parser.add_argument(
        "--alpamayo-model",
        type=str,
        default="/models/Alpamayo-R1-10B",
        help="Path to Alpamayo model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local files (no HuggingFace download)",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Single clip ID from PhysicalAI-AV dataset (optional)",
    )
    parser.add_argument(
        "--clip-ids-file",
        type=str,
        default="/data/physicalai_av/clip_ids.json",
        help="JSON file with clip IDs",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/exp",
        help="Directory to save experiment results",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get clip IDs
    if args.clip_id:
        clip_ids = [args.clip_id]
    else:
        with open(args.clip_ids_file) as f:
            all_clip_ids = json.load(f)
        # Get unique clip IDs
        unique_clip_ids = list(set(all_clip_ids))
        clip_ids = unique_clip_ids[:args.num_samples]

    logger.info(f"Evaluating {len(clip_ids)} samples")

    # Load Alpamayo model
    logger.info(f"Loading Alpamayo model from {args.alpamayo_model}...")
    model = AlpamayoR1.from_pretrained(
        args.alpamayo_model,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    logger.info("Alpamayo model loaded.")

    # Load DFlash draft model
    logger.info(f"Loading DFlash draft model from {args.draft_model}...")
    draft_model = load_dflash_draft_model(
        args.draft_model,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Create accelerator with trajectory token masking
    # Extract trajectory token config from the Alpamayo model
    traj_offset = getattr(model.config, "traj_token_start_idx", None)
    traj_vocab = getattr(model.config, "traj_vocab_size", None)
    # Get mask_token_id from draft model config (critical for correct inference)
    mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft_model.config, "mask_token_id", None)
    if mask_token_id is not None:
        logger.info(f"Using mask_token_id={mask_token_id} from draft model")
    config = DFlashConfig(
        temperature=args.temperature,
        traj_token_offset=traj_offset,
        traj_vocab_size=traj_vocab,
        mask_token_id_override=mask_token_id,
        target_layer_ids=[24, 30, 31, 32, 34],
    )
    accelerator = DFlashAlpamayoAccelerator(
        draft_model=draft_model,
        target_vlm=model.vlm,
        tokenizer=model.tokenizer,
        config=config,
    )

    # Load exact MASK embedding from training if available
    # This eliminates train-inference mismatch from re-computing vocab mean
    draft_path = Path(args.draft_model)
    mask_emb_file = draft_path / "mask_embedding.pt"
    if mask_emb_file.exists():
        logger.info(f"Loading exact mask embedding from {mask_emb_file}")
        mask_emb = torch.load(mask_emb_file, map_location="cuda")
        mask_emb = mask_emb.to(dtype=model.vlm.get_input_embeddings().weight.dtype)
        with torch.no_grad():
            model.vlm.get_input_embeddings().weight[accelerator.mask_token_id] = mask_emb
        logger.info(f"Loaded exact mask embedding for token ID {accelerator.mask_token_id}")
    else:
        logger.warning(f"No mask_embedding.pt found in {draft_path}. Using vocab mean initialization.")

    logger.info("DFlash accelerator created.")

    # Collect results for all samples
    all_results = []
    dflash_times = []
    standard_times = []
    acceptance_rates = []
    acceptance_lengths = []
    speedups = []

    for i, clip_id in enumerate(clip_ids):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {i+1}/{len(clip_ids)}: {clip_id}")
        logger.info("=" * 60)

        try:
            # Load dataset
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

            # Prepare inputs
            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )

            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
            model_inputs = helper.to_device(model_inputs, "cuda")
            tokenized_data = helper.to_device(inputs, "cuda")

            sample_result = {"clip_id": clip_id}

            # Run standard inference if comparing
            if args.compare:
                torch.cuda.manual_seed_all(42)
                import copy
                standard_inputs = copy.deepcopy(model_inputs)

                coc_standard, time_standard, pred_xyz, _ = run_standard_inference(
                    model, standard_inputs, temperature=0.6, max_tokens=args.max_tokens
                )
                standard_times.append(time_standard)

                # Compute minADE
                gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
                pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
                diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
                min_ade = float(diff.min())

                sample_result["standard"] = {
                    "time_ms": time_standard,
                    "min_ade": min_ade,
                }
                logger.info(f"  Standard: {time_standard:.1f}ms, minADE={min_ade:.3f}m")

            # Run DFlash inference
            torch.cuda.manual_seed_all(42)

            if args.verbose:
                # Verbose mode - shows detailed step-by-step information
                torch.cuda.synchronize()
                start = time.perf_counter()

                verbose_result = run_dflash_inference_verbose(
                    model, accelerator, tokenized_data,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )

                torch.cuda.synchronize()
                time_dflash = (time.perf_counter() - start) * 1000

                stats = {
                    "total_tokens": verbose_result["total_tokens"],
                    "total_iterations": verbose_result["total_iterations"],
                    "mean_acceptance_length": verbose_result["mean_acceptance_length"],
                    "acceptance_rate": verbose_result["acceptance_rate"],
                }
                sample_result["verbose_log"] = verbose_result["log"]
            else:
                coc_dflash, time_dflash, stats = run_dflash_inference(
                    model, accelerator, tokenized_data,
                    temperature=args.temperature, max_tokens=args.max_tokens
                )

            dflash_times.append(time_dflash)
            acceptance_rates.append(stats["acceptance_rate"])
            acceptance_lengths.append(stats["mean_acceptance_length"])

            sample_result["dflash"] = {
                "time_ms": time_dflash,
                "total_tokens": stats["total_tokens"],
                "acceptance_rate": stats["acceptance_rate"],
                "mean_acceptance_length": stats["mean_acceptance_length"],
            }

            if not args.verbose:
                logger.info(f"  DFlash:   {time_dflash:.1f}ms, accept_rate={stats['acceptance_rate']:.1%}, accept_len={stats['mean_acceptance_length']:.2f}")

            if args.compare:
                speedup = time_standard / time_dflash if time_dflash > 0 else 0
                speedups.append(speedup)
                sample_result["speedup"] = speedup
                logger.info(f"  Speedup:  {speedup:.2f}x")

            all_results.append(sample_result)

        except Exception as e:
            logger.warning(f"  Error processing {clip_id}: {e}")
            continue

    # Compute aggregate statistics
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)

    n_success = len(dflash_times)
    logger.info(f"Successful samples: {n_success}/{len(clip_ids)}")

    if n_success > 0:
        avg_dflash_time = np.mean(dflash_times)
        avg_acceptance_rate = np.mean(acceptance_rates)
        avg_acceptance_length = np.mean(acceptance_lengths)

        logger.info(f"\nDFlash Metrics:")
        logger.info(f"  Avg time:              {avg_dflash_time:.1f} ms")
        logger.info(f"  Avg acceptance rate:   {avg_acceptance_rate:.1%}")
        logger.info(f"  Avg acceptance length: {avg_acceptance_length:.2f}")

        if args.compare and len(standard_times) > 0:
            avg_standard_time = np.mean(standard_times)
            avg_speedup = np.mean(speedups)
            logger.info(f"\nComparison:")
            logger.info(f"  Avg standard time:     {avg_standard_time:.1f} ms")
            logger.info(f"  Avg speedup:           {avg_speedup:.2f}x")

    # Build final results dict
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "alpamayo_model": args.alpamayo_model,
            "draft_model": args.draft_model,
            "num_samples": len(clip_ids),
            "successful_samples": n_success,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "aggregate": {
            "avg_dflash_time_ms": float(np.mean(dflash_times)) if dflash_times else 0,
            "avg_acceptance_rate": float(np.mean(acceptance_rates)) if acceptance_rates else 0,
            "avg_acceptance_length": float(np.mean(acceptance_lengths)) if acceptance_lengths else 0,
        },
        "samples": all_results,
    }

    if args.compare and standard_times:
        results["aggregate"]["avg_standard_time_ms"] = float(np.mean(standard_times))
        results["aggregate"]["avg_speedup"] = float(np.mean(speedups))

    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f"dflash_eval_{len(clip_ids)}samples_{timestamp_str}.json"

    # For JSON, remove verbose_log from samples (it's saved separately)
    results_for_json = results.copy()
    if args.verbose:
        # Collect all verbose logs
        all_verbose_logs = []
        for sample in results_for_json["samples"]:
            if "verbose_log" in sample:
                all_verbose_logs.append(f"=== Clip: {sample['clip_id']} ===\n{sample['verbose_log']}")
                del sample["verbose_log"]

        # Save verbose log to separate file
        log_file = output_dir / f"dflash_eval_{len(clip_ids)}samples_{timestamp_str}.log"
        with open(log_file, "w") as f:
            f.write("\n\n".join(all_verbose_logs))
        logger.info(f"\nVerbose log saved to: {log_file}")

    with open(result_file, "w") as f:
        json.dump(results_for_json, f, indent=2)
    logger.info(f"Results saved to: {result_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
