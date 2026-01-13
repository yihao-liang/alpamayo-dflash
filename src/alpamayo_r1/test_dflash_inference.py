#!/usr/bin/env python3
"""Test script for DFlash-accelerated Alpamayo inference.

This script compares standard autoregressive inference with DFlash speculative decoding
for Chain-of-Causation (CoC) generation.

Usage:
    python test_dflash_inference.py [--compare] [--draft-model PATH]

Options:
    --compare       Run both standard and DFlash inference for comparison
    --draft-model   Path to DFlash draft model (default: /models/Qwen3-8B-DFlash-b16)
    --output-dir    Directory to save experiment results (default: /exp)
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
    # Get stop token
    stop_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

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


def main():
    parser = argparse.ArgumentParser(description="Test DFlash-accelerated Alpamayo inference")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both standard and DFlash inference for comparison",
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
    )
    accelerator = DFlashAlpamayoAccelerator(
        draft_model=draft_model,
        target_vlm=model.vlm,
        tokenizer=model.tokenizer,
        config=config,
    )
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
    result_file = output_dir / f"dflash_eval_{len(clip_ids)}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {result_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
