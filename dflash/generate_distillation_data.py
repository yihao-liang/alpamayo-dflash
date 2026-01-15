#!/usr/bin/env python3
"""Generate offline distillation data for DFlash training.

Extracts (hidden_state, future_tokens, top_k_logits) from Alpamayo VLM and saves to disk.
This allows fast training with KL loss without loading Alpamayo during training.

Usage:
    python generate_distillation_data.py \
        --cache-dir /data/physicalai_av/hf_cache \
        --output-dir /data/dflash_distillation \
        --num-chunks 400 \
        --stride 1 \
        --top-k-logits 128
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
import physical_ai_av

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DFlash configuration
BLOCK_SIZE = 8  # Smaller blocks for short CoC sequences (avg 14 tokens)

# Standard Qwen3 vocabulary size (before Alpamayo's extensions)
# Tokens >= this value are trajectory tokens and should be masked in training
STANDARD_VOCAB_SIZE = 151936
IGNORE_INDEX = -100  # Standard PyTorch ignore index for CrossEntropyLoss


# For Alpamayo (36 layers) + DFlash (5 draft layers)
NUM_TARGET_LAYERS = 36
NUM_DRAFT_LAYERS = 5

# Default target layers: selected via smart layer selection (cosine similarity + early-exit loss)
DEFAULT_TARGET_LAYER_IDS = [24, 30, 31, 32, 34]


def extract_hidden_states_and_tokens(
    model: AlpamayoR1,
    processor,
    data: dict,
    target_layer_ids: list[int],
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Run Alpamayo forward pass and extract hidden states + generated tokens + logits.

    Args:
        model: Alpamayo model
        processor: Tokenizer processor
        data: Input data dict
        target_layer_ids: List of layer indices to extract hidden states from
        device: Compute device

    Returns:
        hidden_states: (seq_len, num_layers * hidden_dim) - concatenated hidden states
        input_ids: (seq_len,) - full sequence (input + generated)
        generation_start_idx: int - where generation starts
        logits: (seq_len, vocab_size) - target model logits for KL loss
    """
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
    model_inputs = helper.to_device(model_inputs, device)

    input_len = inputs["input_ids"].shape[-1]

    # Run inference to get full sequence
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    # Get the generated token IDs from extra
    # We need to run a forward pass to get hidden states
    vlm = model.vlm

    # Get full input_ids (input + generated CoC + <|cot_end|>)
    # The CoC text is in extra["cot"]
    cot_text = extra["cot"][0][0][0] if extra.get("cot") is not None else ""
    cot_tokens = model.tokenizer(cot_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    # Get <|cot_end|> token ID to append after CoC text
    # This teaches the drafter when to stop generating CoC
    cot_end_token_id = model.tokenizer.convert_tokens_to_ids("<|cot_end|>")
    cot_end_token = torch.tensor([[cot_end_token_id]], dtype=cot_tokens.dtype)

    # Combine input + CoC tokens + <|cot_end|>
    full_input_ids = torch.cat([
        inputs["input_ids"].to(device),
        cot_tokens.to(device),
        cot_end_token.to(device),
    ], dim=-1)

    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = vlm(
            input_ids=full_input_ids,
            pixel_values=inputs.get("pixel_values", None).to(device) if inputs.get("pixel_values") is not None else None,
            image_grid_thw=inputs.get("image_grid_thw", None).to(device) if inputs.get("image_grid_thw") is not None else None,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract hidden states from target layers and concatenate
    # hidden_states[0] = embedding output, hidden_states[i] = output of layer i-1
    # So we need offset=1 to match DFlash's extract_context_feature
    all_hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
    offset = 1  # hidden_states[0] is embedding, layers start at index 1

    # Concatenate hidden states from target layers
    target_hidden_list = []
    for layer_id in target_layer_ids:
        hidden_idx = layer_id + offset
        if hidden_idx < len(all_hidden_states):
            target_hidden_list.append(all_hidden_states[hidden_idx])
        else:
            raise ValueError(
                f"Layer {layer_id} (index {hidden_idx}) out of range. "
                f"Model has {len(all_hidden_states) - 1} layers."
            )

    # (batch, seq, num_layers * hidden)
    target_hidden = torch.cat(target_hidden_list, dim=-1)

    # Get logits for KL loss
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    return target_hidden[0], full_input_ids[0], input_len, logits


def mask_extended_vocab_tokens(
    tokens: torch.Tensor,
    mask_extended: bool = True,
    keep_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Optionally mask tokens outside standard Qwen3 vocabulary with IGNORE_INDEX.

    Alpamayo extends the vocabulary with trajectory tokens (>= 151936).
    Special tokens like <|cot_end|> should NOT be masked so the model learns them.

    Args:
        tokens: Token IDs tensor
        mask_extended: If True, mask extended vocab tokens. If False, keep all tokens.
        keep_token_ids: List of token IDs to NOT mask (e.g., special tokens like <|cot_end|>)

    Returns:
        Labels tensor (with extended tokens replaced by IGNORE_INDEX if mask_extended=True)
    """
    if not mask_extended:
        return tokens.clone()

    labels = tokens.clone()
    mask = labels >= STANDARD_VOCAB_SIZE

    # Don't mask special tokens that we want to train on
    if keep_token_ids is not None:
        for tok_id in keep_token_ids:
            mask = mask & (labels != tok_id)

    if mask.any():
        labels[mask] = IGNORE_INDEX
    return labels


def create_training_blocks(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    generation_start_idx: int,
    block_size: int = BLOCK_SIZE,
    stride: int = 1,
    top_k_logits: int = 128,
    mask_extended_vocab: bool = True,
    keep_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (hidden_state, future_tokens, labels, top_k_logits) tuples with sliding window.

    Args:
        hidden_states: (seq_len, hidden_dim)
        input_ids: (seq_len,)
        logits: (seq_len, vocab_size) - target model logits
        generation_start_idx: where generation starts (we want blocks that predict generated tokens)
        block_size: number of future tokens per block
        stride: step size for sliding window
        top_k_logits: number of top logits to store per position
        mask_extended_vocab: If True, mask trajectory tokens. If False, train on full vocab.
        keep_token_ids: List of token IDs to NOT mask (e.g., special tokens like <|cot_end|>)

    Returns:
        block_hidden: (num_blocks, hidden_dim) - target model hidden states
        block_tokens: (num_blocks, block_size) - actual token IDs (for noise embedding)
        block_labels: (num_blocks, block_size) - labels (optionally masked)
        block_topk_values: (num_blocks, block_size-1, top_k) - top-k logit values
        block_topk_indices: (num_blocks, block_size-1, top_k) - top-k logit indices
    """
    seq_len = hidden_states.shape[0]

    # We want to predict tokens after generation_start_idx
    # Start from positions that have block_size tokens ahead
    start_pos = max(0, generation_start_idx - block_size)  # Include some context before generation
    end_pos = seq_len - block_size

    if end_pos <= start_pos:
        return None, None, None, None, None

    positions = list(range(start_pos, end_pos, stride))

    # When stride > 1, ensure critical positions are included:
    if stride > 1:
        # 1. Prompt/generation boundary position (where inference starts)
        boundary_pos = generation_start_idx - 1
        if boundary_pos >= 0 and boundary_pos not in positions and boundary_pos < end_pos:
            positions.append(boundary_pos)

        # 2. End-of-CoC position (to capture <|cot_end|> token)
        # This is the last position that has block_size tokens ahead
        end_coc_pos = seq_len - block_size - 1
        if end_coc_pos >= 0 and end_coc_pos not in positions and end_coc_pos < end_pos:
            positions.append(end_coc_pos)

        positions.sort()

    block_hidden = []
    block_tokens = []
    block_labels = []
    block_topk_values = []
    block_topk_indices = []

    for pos in positions:
        block_hidden.append(hidden_states[pos])
        tokens = input_ids[pos + 1 : pos + 1 + block_size]
        block_tokens.append(tokens)
        # Optionally mask extended vocab tokens (trajectory tokens > 151936)
        # But keep special tokens like <|cot_end|> so the model learns when to stop
        block_labels.append(mask_extended_vocab_tokens(
            tokens, mask_extended=mask_extended_vocab, keep_token_ids=keep_token_ids
        ))

        # Extract top-k logits for positions 1 to block_size (predicting tokens 2 to block_size)
        # logits[pos] predicts token at pos+1, so logits[pos:pos+block_size-1] predict tokens pos+1 to pos+block_size-1
        # We need logits for predicting tokens at positions pos+2 to pos+block_size (block_size-1 positions)
        block_logits = logits[pos + 1 : pos + block_size]  # (block_size-1, vocab_size)
        topk_vals, topk_inds = block_logits.topk(top_k_logits, dim=-1)
        block_topk_values.append(topk_vals)
        block_topk_indices.append(topk_inds)

    block_hidden = torch.stack(block_hidden)
    block_tokens = torch.stack(block_tokens)
    block_labels = torch.stack(block_labels)
    block_topk_values = torch.stack(block_topk_values)  # (num_blocks, block_size-1, top_k)
    block_topk_indices = torch.stack(block_topk_indices)  # (num_blocks, block_size-1, top_k)

    return block_hidden, block_tokens, block_labels, block_topk_values, block_topk_indices


def main():
    parser = argparse.ArgumentParser(description="Generate offline distillation data")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/models/Alpamayo-R1-10B",
        help="Path to Alpamayo model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/physicalai_av/hf_cache",
        help="HuggingFace cache directory with downloaded chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/dflash_distillation",
        help="Output directory for distillation data",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Number of chunks to process (used if --start-chunk/--end-chunk not set)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=None,
        help="Start chunk index (inclusive). For parallel processing.",
    )
    parser.add_argument(
        "--end-chunk",
        type=int,
        default=None,
        help="End chunk index (exclusive). For parallel processing.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="GPU rank for parallel processing. Uses CUDA device and adds suffix to output files.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window (default: 4)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of blocks per shard file",
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        default=None,
        help="Comma-separated target layer indices (e.g., '24,30,31,32,34'). Default: [24,30,31,32,34]",
    )
    parser.add_argument(
        "--top-k-logits",
        type=int,
        default=128,
        help="Number of top logits to store per position for KL loss (default: 128)",
    )
    parser.add_argument(
        "--full-vocab",
        action="store_true",
        help="Train on full vocabulary including trajectory tokens (don't mask extended vocab)",
    )
    args = parser.parse_args()

    # Parse target layers
    if args.target_layers:
        target_layer_ids = [int(x.strip()) for x in args.target_layers.split(",")]
    else:
        target_layer_ids = DEFAULT_TARGET_LAYER_IDS

    # Determine chunk range
    if args.start_chunk is not None and args.end_chunk is not None:
        chunk_range = range(args.start_chunk, args.end_chunk)
    else:
        chunk_range = range(args.num_chunks)

    # Determine device
    if args.rank is not None:
        # When CUDA_VISIBLE_DEVICES is set, only 1 GPU is visible as device 0
        # rank is only used for file naming, not device selection
        device = "cuda:0"
        torch.cuda.set_device(0)
        rank_suffix = f"_rank{args.rank}"
    else:
        device = "cuda"
        rank_suffix = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"[Rank {args.rank}] Loading Alpamayo model from {args.model_path}...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # Get special token IDs to keep (not mask) during training
    # <|cot_end|> must be trained so the model learns when to stop CoC generation
    cot_end_id = model.tokenizer.convert_tokens_to_ids("<|cot_end|>")
    keep_token_ids = [cot_end_id]
    logger.info(f"[Rank {args.rank}] Keep token IDs (not masked): {keep_token_ids} (<|cot_end|>={cot_end_id})")

    # Load dataset interface
    logger.info(f"[Rank {args.rank}] Loading dataset interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)

    # Get clip IDs from chunks
    logger.info(f"[Rank {args.rank}] Getting clip IDs for chunks {list(chunk_range)}...")
    index_path = hf_hub_download(
        "nvidia/PhysicalAI-Autonomous-Vehicles",
        "clip_index.parquet",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    )
    clip_index = pd.read_parquet(index_path)
    clip_ids = clip_index[clip_index["chunk"].isin(chunk_range)].index.tolist()

    logger.info(f"[Rank {args.rank}] Processing {len(clip_ids)} clips from chunks {list(chunk_range)}")
    logger.info(f"[Rank {args.rank}] Block size: {BLOCK_SIZE}, Stride: {args.stride}")
    logger.info(f"[Rank {args.rank}] Target layers: {target_layer_ids}")
    logger.info(f"[Rank {args.rank}] Top-K logits: {args.top_k_logits}")
    logger.info(f"[Rank {args.rank}] Full vocab (no masking): {args.full_vocab}")

    # Process clips and accumulate blocks
    all_hidden = []
    all_tokens = []
    all_labels = []
    all_topk_values = []
    all_topk_indices = []
    total_blocks = 0
    masked_tokens_count = 0
    shard_idx = 0
    failed_clips = []

    pbar = tqdm(clip_ids, desc=f"[Rank {args.rank}] Processing clips")
    for clip_id in pbar:
        try:
            # Load data
            data = load_physical_aiavdataset(
                clip_id,
                t0_us=5_100_000,
                avdi=avdi,
                maybe_stream=False,
            )

            # Extract hidden states, tokens, and logits
            hidden_states, input_ids, gen_start, logits = extract_hidden_states_and_tokens(
                model, processor, data, target_layer_ids, device=device
            )

            # Create training blocks with top-k logits
            # Pass keep_token_ids so <|cot_end|> is not masked
            block_hidden, block_tokens, block_labels, block_topk_vals, block_topk_inds = create_training_blocks(
                hidden_states.cpu(),
                input_ids.cpu(),
                logits.cpu(),
                gen_start,
                block_size=BLOCK_SIZE,
                stride=args.stride,
                top_k_logits=args.top_k_logits,
                mask_extended_vocab=not args.full_vocab,
                keep_token_ids=keep_token_ids,
            )

            if block_hidden is not None:
                all_hidden.append(block_hidden.half())  # Convert to fp16
                all_tokens.append(block_tokens.int())
                all_labels.append(block_labels.int())
                all_topk_values.append(block_topk_vals.half())  # fp16
                all_topk_indices.append(block_topk_inds.int())  # int32
                total_blocks += block_hidden.shape[0]
                # Count masked tokens (extended vocab)
                masked_tokens_count += (block_labels == IGNORE_INDEX).sum().item()

            pbar.set_postfix({"blocks": total_blocks, "shards": shard_idx})

            # Save shard when enough blocks accumulated
            if total_blocks >= args.shard_size * (shard_idx + 1):
                save_shard(all_hidden, all_tokens, all_labels, all_topk_values, all_topk_indices, output_dir, shard_idx, args, rank_suffix)
                all_hidden = []
                all_labels = []
                all_tokens = []
                all_topk_values = []
                all_topk_indices = []
                shard_idx += 1

        except Exception as e:
            failed_clips.append({"clip_id": clip_id, "error": str(e)})
            logger.warning(f"[Rank {args.rank}] Failed to process {clip_id}: {e}")

    # Save remaining blocks
    if all_hidden:
        save_shard(all_hidden, all_tokens, all_labels, all_topk_values, all_topk_indices, output_dir, shard_idx, args, rank_suffix)
        shard_idx += 1

    # Save metadata
    total_tokens = total_blocks * BLOCK_SIZE
    metadata = {
        "rank": args.rank,
        "chunk_range": list(chunk_range),
        "num_shards": shard_idx,
        "total_blocks": total_blocks,
        "block_size": BLOCK_SIZE,
        "stride": args.stride,
        "target_layer_ids": target_layer_ids,
        "num_target_layers": NUM_TARGET_LAYERS,
        "num_draft_layers": NUM_DRAFT_LAYERS,
        "top_k_logits": args.top_k_logits,
        "full_vocab": args.full_vocab,
        "num_clips": len(clip_ids),
        "failed_clips": len(failed_clips),
        "standard_vocab_size": STANDARD_VOCAB_SIZE,
        "masked_tokens": masked_tokens_count,
        "masked_token_ratio": masked_tokens_count / total_tokens if total_tokens > 0 else 0,
    }

    metadata_filename = f"metadata{rank_suffix}.json"
    with open(output_dir / metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)

    if failed_clips:
        failed_filename = f"failed_clips{rank_suffix}.json"
        with open(output_dir / failed_filename, "w") as f:
            json.dump(failed_clips, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"[Rank {args.rank}] Generation complete!")
    logger.info(f"  Chunks: {list(chunk_range)}")
    logger.info(f"  Total blocks: {total_blocks:,}")
    logger.info(f"  Shards: {shard_idx}")
    logger.info(f"  Failed clips: {len(failed_clips)}")
    if total_tokens > 0:
        logger.info(f"  Masked tokens: {masked_tokens_count:,} ({100*masked_tokens_count/total_tokens:.2f}%)")
    logger.info(f"  Output: {output_dir}")


def save_shard(all_hidden, all_tokens, all_labels, all_topk_values, all_topk_indices, output_dir, shard_idx, args, rank_suffix=""):
    """Save accumulated blocks to a shard file.

    Each shard contains:
        - target_hidden: Hidden states from Alpamayo (for conditioning drafter)
        - future_tokens: Actual token IDs (for creating noise embeddings)
        - labels: Masked labels for loss computation (extended vocab tokens → -100)
        - topk_values: Top-k logit values for KL loss (fp16)
        - topk_indices: Top-k logit indices for KL loss (int32)
    """
    hidden = torch.cat(all_hidden, dim=0)
    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.cat(all_labels, dim=0)
    topk_values = torch.cat(all_topk_values, dim=0)
    topk_indices = torch.cat(all_topk_indices, dim=0)

    shard_path = output_dir / f"shard{rank_suffix}_{shard_idx:04d}.pt"
    torch.save({
        "target_hidden": hidden,      # (num_blocks, hidden_dim) fp16
        "future_tokens": tokens,      # (num_blocks, block_size) int32
        "labels": labels,             # (num_blocks, block_size) int32, with -100 for masked
        "topk_values": topk_values,   # (num_blocks, block_size-1, top_k) fp16
        "topk_indices": topk_indices, # (num_blocks, block_size-1, top_k) int32
    }, shard_path)

    # Count masked tokens in this shard
    masked_in_shard = (labels == IGNORE_INDEX).sum().item()
    total_in_shard = labels.numel()
    logger.info(
        f"[Rank {args.rank}] Saved shard {shard_idx}: {hidden.shape[0]} blocks, "
        f"{masked_in_shard}/{total_in_shard} masked tokens, {shard_path}"
    )


if __name__ == "__main__":
    main()
