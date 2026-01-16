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
    --draft-model   Path to DFlash draft model (default: /exp/dflash_train_0115_163242/best)
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
) -> tuple[str, dict, torch.Tensor, torch.Tensor]:
    """Run standard autoregressive inference with timing breakdown.

    Note: This measures wall-clock time for the full pipeline (VLM + diffusion).
    The internal timing breakdown (prefill/decode) is from vlm.generate() only.

    Returns:
        Tuple of (coc_text, timing_dict, pred_xyz, pred_rot).
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
    total_time_ms = (time.perf_counter() - start) * 1000

    coc_text = extra["cot"][0, 0, 0] if extra["cot"].ndim == 3 else extra["cot"][0][0]

    # Get internal VLM timing breakdown
    vlm_timing = extra.get("timing", {})
    if not vlm_timing:
        logger.warning("No timing info returned from model - using wall clock time only")

    # Build timing dict with wall-clock total and VLM breakdown
    timing = {
        "total_time_ms": total_time_ms,  # Wall-clock: VLM + diffusion
        "vlm_total_ms": vlm_timing.get("total_time_ms", 0),  # VLM generation only
        "vlm_prefill_ms": vlm_timing.get("prefill_time_ms", 0),
        "vlm_decode_ms": vlm_timing.get("decode_time_ms", 0),
        "num_decode_steps": vlm_timing.get("num_decode_steps", 0),
        "diffusion_time_ms": vlm_timing.get("diffusion_time_ms", 0),  # Trajectory diffusion
    }

    return coc_text, timing, pred_xyz, pred_rot


def run_dflash_inference(
    model: AlpamayoR1,
    accelerator: DFlashAlpamayoAccelerator,
    tokenized_data: dict,
    temperature: float = 0.0,
    max_tokens: int = 256,
    enable_detailed_timing: bool = False,
) -> tuple[str, float, dict]:
    """Run DFlash-accelerated inference for CoC generation only.

    Note: This only generates the CoC text, not the full trajectory.
    The trajectory generation would use the standard diffusion sampling.

    Args:
        enable_detailed_timing: If True, measure detailed timing breakdown
            (draft, verify, sample, cache) using CUDA events. Default False for
            maximum speed.

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
            enable_detailed_timing=enable_detailed_timing,
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
        "tokens_per_second": stats.total_tokens / (stats.decode_time_ms / 1000) if stats.decode_time_ms > 0 else 0,
        # Detailed timing breakdown
        "draft_time_ms": stats.draft_time_ms,
        "verify_time_ms": stats.verify_time_ms,
        "sample_time_ms": stats.sample_time_ms,
        "cache_time_ms": stats.cache_time_ms,
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
    log(f"Target layer IDs: {target_layer_ids}")
    log(f"Mask token ID: {mask_token_id}")
    log(f"Max new tokens: {max_tokens}")
    log()

    # ====== PREFILL ======
    log(">>> PREFILL STAGE")
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefill_output = accelerator.target_vlm(
            input_ids=input_ids,
            pixel_values=tokenized_data.get("pixel_values"),
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

    torch.cuda.synchronize()
    prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

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

    # Debug: show prefill context stats
    prefill_norm = target_hidden.norm().item()
    prefill_mean = target_hidden.mean().item()
    prefill_std = target_hidden.std().item()
    log(f"[Prefill context: norm={prefill_norm:.2f}, mean={prefill_mean:.4f}, std={prefill_std:.4f}]")
    log(f"Input length: {num_input_tokens}")
    log(f"Prefill time: {prefill_time_ms:.1f} ms")
    log()

    # ====== DECODE LOOP ======
    torch.cuda.synchronize()
    decode_start = time.perf_counter()

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

        # Heuristic: if first_in_block is "." (13), directly append <|cot_end|> and stop
        # No need to run draft model - we know the answer
        # Note: don't add a step to `steps` list - this isn't a "real" speculative decoding step
        period_token_id = 13
        if first_in_block == period_token_id:
            log(f"  [Heuristic: '.' detected, directly appending <|cot_end|> and stopping]")
            output_ids[:, start_pos + 1] = stop_token_id
            start_pos += 2  # Move past "." and "<|cot_end|>"
            break  # Stop generation

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

        # Debug: Check context features going into draft model
        ctx_norm = current_context.norm().item()
        ctx_mean = current_context.mean().item()
        ctx_std = current_context.std().item()
        log(f"  [Draft input context: norm={ctx_norm:.2f}, mean={ctx_mean:.4f}, std={ctx_std:.4f}]")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            draft_hidden = accelerator.draft_model(
                target_hidden=current_context,
                noise_embedding=noise_embedding,
                position_ids=reset_position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=False,
            )

        # Debug: Check draft hidden states
        draft_hidden_norm = draft_hidden.norm().item()
        draft_hidden_mean = draft_hidden.mean().item()
        draft_hidden_std = draft_hidden.std().item()
        log(f"  [Draft output hidden: norm={draft_hidden_norm:.2f}, mean={draft_hidden_mean:.4f}, std={draft_hidden_std:.4f}]")

        # Get draft predictions
        draft_logits = accelerator._lm_head(draft_hidden[:, 1:, :])
        draft_tokens = sample_tokens(draft_logits, temperature, accelerator._logits_processor)
        block_output_ids[:, 1:] = draft_tokens

        # Heuristic: if draft predicts <|cot_end|> without preceding ".", replace it with "."
        # This fixes cases where draft skips "." and jumps to <|cot_end|>
        # The next step will then have first_in_block="." and directly append <|cot_end|>
        draft_list = block_output_ids[0, 1:].tolist()
        for i, tok in enumerate(draft_list):
            if tok == stop_token_id:  # Found <|cot_end|>
                # Check if previous token is "."
                prev_tok = block_output_ids[0, i].item()  # i because block_output_ids includes first_in_block at 0
                if prev_tok != period_token_id:
                    # Replace <|cot_end|> with "." - next step will add <|cot_end|> after "."
                    block_output_ids[0, i + 1] = period_token_id
                    log(f"  [Heuristic: replacing <|cot_end|> with '.' at position {i+1}]")
                break

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

        # Check stop condition - look for stop token ANYWHERE in accepted sequence
        # Not just at the end, since draft might predict <|cot_end|> mid-block
        hit_stop_token = False
        stop_position = None
        for i, tok_id in enumerate(accepted_ids):
            if tok_id == stop_token_id:
                hit_stop_token = True
                stop_position = i
                break

        # If stop token found mid-sequence, truncate acceptance
        # Note: accepted_ids has acceptance_length + 2 elements (first_in_block + draft + posterior)
        # But actual NEW tokens per step is acceptance_length + 1 (first_in_block was from previous step)
        actual_acceptance = acceptance_length + 1  # default: all new tokens accepted
        draft_matches = acceptance_length  # for acceptance rate calculation
        if hit_stop_token and stop_position is not None:
            # stop_position is index in accepted_ids array
            # Index 0 = first_in_block (not new), indices 1+ = new tokens
            # New tokens = stop_position (indices 1 through stop_position, not counting index 0)
            actual_acceptance = stop_position
            # Draft matches: if stop at index i, we accepted draft positions 1 through min(i, 7)
            # (index 8 would be posterior, indices 1-7 are draft tokens)
            draft_matches = min(stop_position, block_size - 1)
            if stop_position < len(accepted_ids) - 1:
                log(f"  [Stop token found at position {stop_position}, truncating acceptance]")
                # Clear tokens after stop position
                output_ids[:, start_pos + stop_position + 1:] = mask_token_id
                # Update accepted_ids for logging
                accepted_ids = accepted_ids[:stop_position + 1]
                accepted_strs = accepted_strs[:stop_position + 1]

        # Record step info
        step_info = {
            "step": step_num,
            "position": start_pos,
            "first_token": {"id": first_in_block, "text": first_in_block_str},
            "draft_tokens": {"ids": draft_token_ids, "text": draft_token_strs},
            "target_tokens": {"ids": posterior_ids, "text": posterior_strs},
            "matches": match_list,
            "acceptance_length": actual_acceptance,
            "draft_matches": draft_matches,  # for acceptance rate calculation
            "accepted_tokens": {"ids": accepted_ids, "text": accepted_strs},
        }
        steps.append(step_info)

        # Stop immediately if we hit the stop token
        if hit_stop_token:
            log(f"\n  >>> STOP TOKEN (<|cot_end|>) accepted, stopping")
            start_pos += actual_acceptance
            break

        # Update position
        start_pos += acceptance_length + 1

        # Update cache - crop to keep only valid positions
        past_key_values_target.crop(start_pos)

        # Extract hidden state for next iteration's context.
        #
        # Key insight: In verify_output, the hidden at index `acceptance_length` was computed
        # using ONLY accepted tokens [known, draft_1, ..., draft_{acceptance_length}].
        # Due to causal attention, this hidden state does NOT see the rejected draft tokens
        # at later positions. Therefore, it's CORRECT and we can use it directly.
        #
        # - Full match (acceptance_length = block_size - 1): use hidden at index -1
        # - Rejection (acceptance_length < block_size - 1): use hidden at index acceptance_length
        # Both cases unify to: hidden at index acceptance_length
        new_hidden = extract_context_feature(verify_output.hidden_states, target_layer_ids)
        target_hidden = new_hidden[:, acceptance_length : acceptance_length + 1, :]

        # Debug: show hidden state statistics to diagnose feature quality
        hidden_norm = target_hidden.norm().item()
        hidden_mean = target_hidden.mean().item()
        hidden_std = target_hidden.std().item()
        log(f"  [Hidden: from verify_output at index {acceptance_length}, "
            f"norm={hidden_norm:.2f}, mean={hidden_mean:.4f}, std={hidden_std:.4f}]")

        if step_num >= 50:  # Safety limit for verbose mode
            log(f"\n  >>> REACHED MAX STEPS (50)")
            break

    # End decode timing
    torch.cuda.synchronize()
    decode_time_ms = (time.perf_counter() - decode_start) * 1000

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
    draft_matches_list = [s["draft_matches"] for s in steps]

    mean_acceptance = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0
    total_drafted = total_iterations * (block_size - 1)
    total_accepted = sum(draft_matches_list)
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    tokens_per_second = total_tokens / (decode_time_ms / 1000) if decode_time_ms > 0 else 0

    log(f"Total tokens generated: {total_tokens}")
    log(f"Total iterations: {total_iterations}")
    log(f"Prefill time: {prefill_time_ms:.1f} ms")
    log(f"Decode time:  {decode_time_ms:.1f} ms")
    log(f"Tokens/sec:   {tokens_per_second:.1f}")
    log(f"Mean acceptance length: {mean_acceptance:.2f}")
    log(f"Acceptance rate: {acceptance_rate:.1%}")
    log()

    # Per-step summary
    log("Per-step acceptance:")
    for s in steps:
        log(f"  Step {s['step']:2d}: accepted {s['acceptance_length']} tokens - "
            f"first='{s['first_token']['text']}' ({s['first_token']['id']})")

    # Decode final output - only show the CoT portion (not the full prompt)
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # Extract just the CoT part for display
    cot_start_marker = "<|cot_start|>"
    cot_end_marker = "<|cot_end|>"
    if cot_start_marker in generated_text:
        cot_start_idx = generated_text.find(cot_start_marker)
        cot_end_idx = generated_text.find(cot_end_marker)
        if cot_end_idx > cot_start_idx:
            cot_text = generated_text[cot_start_idx:cot_end_idx + len(cot_end_marker)]
        else:
            cot_text = generated_text[cot_start_idx:]
        log(f"\nGenerated CoT:\n{cot_text}")
    else:
        # Fallback: show last 200 chars
        log(f"\nGenerated text (last 200 chars):\n...{generated_text[-200:]}")

    return {
        "steps": steps,
        "total_tokens": total_tokens,
        "total_iterations": total_iterations,
        "mean_acceptance_length": mean_acceptance,
        "acceptance_rate": acceptance_rate,
        "prefill_time_ms": prefill_time_ms,
        "decode_time_ms": decode_time_ms,
        "tokens_per_second": tokens_per_second,
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
        default="/exp/dflash_train_0115_163242/best",
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
    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help="Enable detailed timing breakdown (draft/verify/sample/cache). Adds slight overhead.",
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
    prefill_times = []
    decode_times = []
    tokens_per_sec = []
    # Detailed DFlash timing
    draft_times = []
    verify_times = []
    sample_times = []
    cache_times = []
    # Standard timing
    standard_times = []
    standard_prefill_times = []
    standard_decode_times = []
    standard_tokens_per_sec = []
    diffusion_times = []
    acceptance_rates = []
    acceptance_lengths = []
    speedups = []
    decode_speedups = []

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

                # Run standard inference with timing breakdown
                # Note: total_time_ms = wall clock (VLM + diffusion)
                #       vlm_* times = VLM generation only (for fair comparison)
                coc_standard, timing, pred_xyz, _ = run_standard_inference(
                    model, standard_inputs, temperature=0.6, max_tokens=args.max_tokens
                )

                time_standard = timing.get("total_time_ms", 0)  # Wall clock (includes diffusion)
                vlm_total_ms = timing.get("vlm_total_ms", 0)    # VLM only (for comparison)
                vlm_prefill_ms = timing.get("vlm_prefill_ms", 0)
                vlm_decode_ms = timing.get("vlm_decode_ms", 0)
                diffusion_ms = timing.get("diffusion_time_ms", 0)
                num_tokens = timing.get("num_decode_steps", 0)
                tps = num_tokens / (vlm_decode_ms / 1000) if vlm_decode_ms > 0 else 0

                standard_times.append(vlm_total_ms)  # Use VLM-only time for fair comparison
                standard_prefill_times.append(vlm_prefill_ms)
                standard_decode_times.append(vlm_decode_ms)
                standard_tokens_per_sec.append(tps)
                diffusion_times.append(diffusion_ms)

                # Compute minADE
                gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
                pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
                diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
                min_ade = float(diff.min())

                sample_result["standard"] = {
                    "time_ms": time_standard,  # Wall clock (VLM + diffusion)
                    "vlm_total_ms": vlm_total_ms,
                    "vlm_prefill_ms": vlm_prefill_ms,
                    "vlm_decode_ms": vlm_decode_ms,
                    "diffusion_ms": diffusion_ms,
                    "tokens_per_second": tps,
                    "total_tokens": num_tokens,
                    "min_ade": min_ade,
                }
                logger.info(f"  Standard: {vlm_total_ms:.1f}ms VLM (prefill={vlm_prefill_ms:.1f}ms, decode={vlm_decode_ms:.1f}ms) + {diffusion_ms:.1f}ms diffusion")
                logger.info(f"            tokens={num_tokens}, tok/s={tps:.1f}, minADE={min_ade:.3f}m")

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
                    "prefill_time_ms": verbose_result["prefill_time_ms"],
                    "decode_time_ms": verbose_result["decode_time_ms"],
                    "tokens_per_second": verbose_result["tokens_per_second"],
                }
                sample_result["verbose_log"] = verbose_result["log"]
            else:
                coc_dflash, time_dflash, stats = run_dflash_inference(
                    model, accelerator, tokenized_data,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    enable_detailed_timing=args.detailed_timing,
                )

            dflash_times.append(time_dflash)
            acceptance_rates.append(stats["acceptance_rate"])
            acceptance_lengths.append(stats["mean_acceptance_length"])
            if "prefill_time_ms" in stats:
                prefill_times.append(stats["prefill_time_ms"])
                decode_times.append(stats["decode_time_ms"])
                tokens_per_sec.append(stats.get("tokens_per_second", 0))
            # Collect detailed timing (only if enabled - values will be non-zero)
            if stats.get("draft_time_ms", 0) > 0:
                draft_times.append(stats["draft_time_ms"])
                verify_times.append(stats["verify_time_ms"])
                sample_times.append(stats["sample_time_ms"])
                cache_times.append(stats["cache_time_ms"])

            sample_result["dflash"] = {
                "time_ms": time_dflash,
                "total_tokens": stats["total_tokens"],
                "acceptance_rate": stats["acceptance_rate"],
                "mean_acceptance_length": stats["mean_acceptance_length"],
                "prefill_time_ms": stats.get("prefill_time_ms", 0),
                "decode_time_ms": stats.get("decode_time_ms", 0),
                "tokens_per_second": stats.get("tokens_per_second", 0),
            }

            if not args.verbose:
                # Show diffusion time if available (same as standard since DFlash only accelerates VLM)
                if args.compare:
                    logger.info(f"  DFlash:   {time_dflash:.1f}ms (prefill={stats.get('prefill_time_ms', 0):.1f}ms, decode={stats.get('decode_time_ms', 0):.1f}ms) + {diffusion_ms:.1f}ms diffusion")
                else:
                    logger.info(f"  DFlash:   {time_dflash:.1f}ms (prefill={stats.get('prefill_time_ms', 0):.1f}ms, decode={stats.get('decode_time_ms', 0):.1f}ms)")
                logger.info(f"            accept_rate={stats['acceptance_rate']:.1%}, accept_len={stats['mean_acceptance_length']:.2f}, tok/s={stats.get('tokens_per_second', 0):.1f}")
                # Detailed decode breakdown (only show if detailed timing enabled)
                draft_ms = stats.get('draft_time_ms', 0)
                if draft_ms > 0:
                    verify_ms = stats.get('verify_time_ms', 0)
                    sample_ms = stats.get('sample_time_ms', 0)
                    cache_ms = stats.get('cache_time_ms', 0)
                    logger.info(f"            decode breakdown: draft={draft_ms:.1f}ms, verify={verify_ms:.1f}ms, sample={sample_ms:.1f}ms, cache={cache_ms:.1f}ms")

            if args.compare:
                # Compare VLM-only times (fair comparison - both doing CoC generation)
                vlm_total = sample_result["standard"].get("vlm_total_ms", 0)
                speedup = vlm_total / time_dflash if time_dflash > 0 else 0
                speedups.append(speedup)
                # Compute decode-only speedup (more meaningful for speculative decoding)
                dflash_decode = stats.get("decode_time_ms", 0)
                standard_decode = sample_result["standard"].get("vlm_decode_ms", 0)
                if dflash_decode > 0 and standard_decode > 0:
                    decode_speedup = standard_decode / dflash_decode
                else:
                    decode_speedup = float('nan')
                    logger.warning(f"  Cannot compute decode speedup: standard_decode={standard_decode}, dflash_decode={dflash_decode}")
                decode_speedups.append(decode_speedup)
                sample_result["speedup"] = speedup
                sample_result["decode_speedup"] = decode_speedup
                logger.info(f"  Speedup:  {speedup:.2f}x (VLM only), {decode_speedup:.2f}x (decode only)")

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
        logger.info(f"  Avg total time:        {avg_dflash_time:.1f} ms")
        if prefill_times:
            avg_prefill = np.mean(prefill_times)
            avg_decode = np.mean(decode_times)
            avg_tps = np.mean(tokens_per_sec)
            logger.info(f"  Avg prefill time:      {avg_prefill:.1f} ms")
            logger.info(f"  Avg decode time:       {avg_decode:.1f} ms")
            logger.info(f"  Avg tokens/sec:        {avg_tps:.1f}")
        # Detailed decode breakdown (only show if detailed timing was enabled)
        if draft_times and sum(draft_times) > 0:
            avg_draft = np.mean(draft_times)
            avg_verify = np.mean(verify_times)
            avg_sample = np.mean(sample_times)
            avg_cache = np.mean(cache_times)
            logger.info(f"  Decode breakdown:")
            logger.info(f"    - Draft model:       {avg_draft:.1f} ms ({100*avg_draft/avg_decode:.1f}%)")
            logger.info(f"    - Target verify:     {avg_verify:.1f} ms ({100*avg_verify/avg_decode:.1f}%)")
            logger.info(f"    - Token sampling:    {avg_sample:.1f} ms ({100*avg_sample/avg_decode:.1f}%)")
            logger.info(f"    - KV cache ops:      {avg_cache:.1f} ms ({100*avg_cache/avg_decode:.1f}%)")
            other_time = avg_decode - avg_draft - avg_verify - avg_sample - avg_cache
            logger.info(f"    - Other overhead:    {other_time:.1f} ms ({100*other_time/avg_decode:.1f}%)")
        logger.info(f"  Avg acceptance rate:   {avg_acceptance_rate:.1%}")
        logger.info(f"  Avg acceptance length: {avg_acceptance_length:.2f}")
        # Show diffusion time if available (same for DFlash and Standard)
        if diffusion_times:
            avg_diffusion = np.mean(diffusion_times)
            logger.info(f"  Avg diffusion time:    {avg_diffusion:.1f} ms (same as standard)")

        if args.compare and len(standard_times) > 0:
            avg_standard_time = np.mean(standard_times)  # Wall clock (VLM + diffusion)
            avg_standard_prefill = np.mean(standard_prefill_times)  # VLM prefill only
            avg_standard_decode = np.mean(standard_decode_times)    # VLM decode only
            avg_standard_tps = np.mean(standard_tokens_per_sec)
            avg_speedup = np.mean(speedups)
            # Filter out NaN values for decode speedup
            valid_decode_speedups = [s for s in decode_speedups if not np.isnan(s)]
            avg_decode_speedup = np.mean(valid_decode_speedups) if valid_decode_speedups else float('nan')

            logger.info(f"\nStandard (autoregressive) - VLM generation only:")
            logger.info(f"  Avg VLM total:         {avg_standard_time:.1f} ms")
            logger.info(f"  Avg VLM prefill:       {avg_standard_prefill:.1f} ms")
            logger.info(f"  Avg VLM decode:        {avg_standard_decode:.1f} ms")
            logger.info(f"  Avg tokens/sec:        {avg_standard_tps:.1f}")
            if diffusion_times:
                avg_diffusion = np.mean(diffusion_times)
                logger.info(f"  Avg diffusion time:    {avg_diffusion:.1f} ms")

            logger.info(f"\nSpeedup (CoC generation only):")
            logger.info(f"  VLM total speedup:     {avg_speedup:.2f}x")
            logger.info(f"  VLM decode speedup:    {avg_decode_speedup:.2f}x  <-- speculative decoding benefit")

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
            "avg_prefill_time_ms": float(np.mean(prefill_times)) if prefill_times else 0,
            "avg_decode_time_ms": float(np.mean(decode_times)) if decode_times else 0,
            "avg_tokens_per_second": float(np.mean(tokens_per_sec)) if tokens_per_sec else 0,
            "avg_acceptance_rate": float(np.mean(acceptance_rates)) if acceptance_rates else 0,
            "avg_acceptance_length": float(np.mean(acceptance_lengths)) if acceptance_lengths else 0,
        },
        "samples": all_results,
    }

    if args.compare and standard_times:
        results["aggregate"]["avg_standard_time_ms"] = float(np.mean(standard_times))
        results["aggregate"]["avg_standard_prefill_ms"] = float(np.mean(standard_prefill_times))
        results["aggregate"]["avg_standard_decode_ms"] = float(np.mean(standard_decode_times))
        results["aggregate"]["avg_standard_tokens_per_sec"] = float(np.mean(standard_tokens_per_sec))
        results["aggregate"]["avg_speedup"] = float(np.mean(speedups))
        results["aggregate"]["avg_decode_speedup"] = float(np.mean(decode_speedups))
        if diffusion_times:
            results["aggregate"]["avg_diffusion_time_ms"] = float(np.mean(diffusion_times))

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
