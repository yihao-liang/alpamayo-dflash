#!/usr/bin/env python3
"""Compute cosine similarity of each layer to final layer.

Usage:
    # Single GPU
    python compute_cosine_similarity.py --num-samples 32

    # Multi-GPU (8 GPUs)
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python compute_cosine_similarity.py \
            --rank $i --world-size 8 --num-samples 32 &
    done
    wait
    python compute_cosine_similarity.py --aggregate --plot

Output:
    - cosine_similarity.json: {layer_id: avg_similarity}
    - cosine_similarity.png: visualization
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_hidden_states(model, processor, data, device):
    """Forward pass, return all hidden states and text start index."""
    from alpamayo_r1 import helper

    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    input_len = inputs["input_ids"].shape[-1]

    # Get CoC tokens
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True,
        )

    cot_text = extra.get("cot", [[[""]]])[0][0][0]
    cot_tokens = model.tokenizer(cot_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    full_input_ids = torch.cat([inputs["input_ids"].to(device), cot_tokens.to(device)], dim=-1)

    with torch.no_grad():
        outputs = model.vlm(
            input_ids=full_input_ids,
            pixel_values=inputs.get("pixel_values").to(device) if inputs.get("pixel_values") is not None else None,
            image_grid_thw=inputs.get("image_grid_thw").to(device) if inputs.get("image_grid_thw") is not None else None,
            output_hidden_states=True, return_dict=True,
        )

    hidden_states = [h[0].cpu() for h in outputs.hidden_states]
    return hidden_states, input_len


def compute_cosine_similarity(hidden_states, text_start_idx):
    """Compute cosine similarity of each layer to final layer."""
    num_layers = len(hidden_states) - 1

    final_hidden = hidden_states[-1][text_start_idx:].float()
    final_mean = F.normalize(final_hidden.mean(dim=0), p=2, dim=0)

    similarities = {}
    for layer_id in range(num_layers):
        layer_hidden = hidden_states[layer_id + 1][text_start_idx:].float()
        layer_mean = F.normalize(layer_hidden.mean(dim=0), p=2, dim=0)
        sim = torch.dot(layer_mean, final_mean).item()
        similarities[layer_id] = max(-1.0, min(1.0, sim))

    return similarities


def plot_results(similarities, output_path):
    """Plot cosine similarity curve."""
    import matplotlib.pyplot as plt

    layers = sorted(similarities.keys())
    values = [similarities[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, values, 'b-o', linewidth=2, markersize=5)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cosine Similarity to Final Layer')
    ax.set_title('Layer-wise Cosine Similarity')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Mark regions
    num_layers = len(layers)
    ax.axvline(x=num_layers * 0.3, color='orange', linestyle='--', alpha=0.7, label='30%')
    ax.axvline(x=num_layers * 0.7, color='red', linestyle='--', alpha=0.7, label='70%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/Alpamayo-R1-10B")
    parser.add_argument("--cache-dir", default="/data/physicalai_av/hf_cache")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", default="layer_selection")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate mode
    if args.aggregate:
        all_sims = []
        for f in sorted(output_dir.glob("cosine_rank*.json")):
            with open(f) as fp:
                all_sims.extend(json.load(fp)["samples"])

        avg = {}
        num_layers = len(all_sims[0])
        for l in range(num_layers):
            avg[l] = np.mean([s[str(l)] for s in all_sims])

        with open(output_dir / "cosine_similarity.json", "w") as f:
            json.dump(avg, f, indent=2)
        logger.info(f"Aggregated {len(all_sims)} samples -> cosine_similarity.json")

        if args.plot:
            plot_results(avg, output_dir / "cosine_similarity.png")
        return

    # Compute mode
    device = f"cuda:{args.rank}" if args.world_size > 1 else "cuda"
    if args.world_size > 1:
        torch.cuda.set_device(args.rank)

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    import physical_ai_av
    import pandas as pd
    from huggingface_hub import hf_hub_download

    logger.info(f"[Rank {args.rank}] Loading model...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.cache_dir)
    index_path = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles", "clip_index.parquet",
                                  repo_type="dataset", cache_dir=args.cache_dir)
    clip_ids = pd.read_parquet(index_path).query("chunk < 10").index.tolist()
    clip_ids = clip_ids[args.rank::args.world_size]

    samples = []
    for clip_id in tqdm(clip_ids[:args.num_samples], desc=f"[Rank {args.rank}]"):
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi, maybe_stream=False)
            hidden_states, text_start = get_hidden_states(model, processor, data, device)
            sim = compute_cosine_similarity(hidden_states, text_start)
            samples.append(sim)
        except Exception as e:
            logger.warning(f"Failed {clip_id}: {e}")

    with open(output_dir / f"cosine_rank{args.rank}.json", "w") as f:
        json.dump({"samples": samples}, f)
    logger.info(f"[Rank {args.rank}] Saved {len(samples)} samples")


if __name__ == "__main__":
    main()
