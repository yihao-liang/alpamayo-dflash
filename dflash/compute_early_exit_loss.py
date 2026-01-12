#!/usr/bin/env python3
"""Compute early-exit cross-entropy loss for specified layers.

Usage:
    # Single GPU
    python compute_early_exit_loss.py --layers "24,30,31,32,34" --num-samples 32

    # Multi-GPU (8 GPUs)
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python compute_early_exit_loss.py \
            --rank $i --world-size 8 --num-samples 32 \
            --layers "0,16,20,24,28,29,30,31,32,34,35" &
    done
    wait
    python compute_early_exit_loss.py --aggregate --plot --layers "0,16,20,24,28,29,30,31,32,34,35"

Output:
    - early_exit_loss.json: {layer_id: avg_loss}
    - early_exit_loss.png: visualization
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_hidden_states_and_labels(model, processor, data, device):
    """Forward pass, return hidden states and labels."""
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
    labels = full_input_ids[0].cpu()
    return hidden_states, labels


def compute_early_exit_loss(model, hidden_state, labels, device):
    """Compute CE loss: hidden -> final_norm -> lm_head -> loss."""
    final_norm = model.vlm.model.language_model.norm
    lm_head = model.vlm.lm_head
    loss_fn = torch.nn.CrossEntropyLoss()

    h = hidden_state.to(device).unsqueeze(0)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h_normed = final_norm(h)
        logits = lm_head(h_normed)

    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[1:].to(device)

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss.item()


def plot_results(losses, output_path):
    """Plot early-exit loss bar chart."""
    import matplotlib.pyplot as plt

    layers = sorted(losses.keys())
    values = [losses[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(layers, values, color='steelblue', edgecolor='navy')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Early-Exit Loss (final_norm -> lm_head)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/Alpamayo-R1-10B")
    parser.add_argument("--cache-dir", default="/data/physicalai_av/hf_cache")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--layers", type=str, required=False, help="Comma-separated layer indices")
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
        all_losses = {}
        for f in sorted(output_dir.glob("loss_rank*.json")):
            with open(f) as fp:
                data = json.load(fp)
            for layer, vals in data["losses"].items():
                layer = int(layer)
                if layer not in all_losses:
                    all_losses[layer] = []
                all_losses[layer].extend(vals)

        avg = {l: np.mean(v) for l, v in all_losses.items()}

        with open(output_dir / "early_exit_loss.json", "w") as f:
            json.dump(avg, f, indent=2)
        logger.info(f"Aggregated -> early_exit_loss.json")

        if args.plot:
            plot_results(avg, output_dir / "early_exit_loss.png")
        return

    # Compute mode
    if not args.layers:
        parser.error("--layers required for compute mode")

    layers = [int(x.strip()) for x in args.layers.split(",")]
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

    losses = {l: [] for l in layers}
    for clip_id in tqdm(clip_ids[:args.num_samples], desc=f"[Rank {args.rank}]"):
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=5_100_000, avdi=avdi, maybe_stream=False)
            hidden_states, labels = get_hidden_states_and_labels(model, processor, data, device)
            for layer_id in layers:
                h = hidden_states[layer_id + 1]
                loss = compute_early_exit_loss(model, h, labels, device)
                losses[layer_id].append(loss)
        except Exception as e:
            logger.warning(f"Failed {clip_id}: {e}")

    with open(output_dir / f"loss_rank{args.rank}.json", "w") as f:
        json.dump({"losses": losses}, f)
    logger.info(f"[Rank {args.rank}] Saved {len(losses[layers[0]])} samples")


if __name__ == "__main__":
    main()
