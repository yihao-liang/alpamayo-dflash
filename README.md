# DFlash + Alpamayo Integration

Notes on integrating DFlash speculative decoding with Alpamayo CoC generation.

## Goal

Trying to speed up Alpamayo's Chain-of-Causation text generation using DFlash block diffusion. Only targeting the text part, not action decoding.

## Progress

**Distillation training complete. Evaluating acceptance rates.**

## Results

| What | Acceptance Rate | Speedup | Notes |
|------|-----------------|---------|-------|
| Pre-trained DFlash   | 0.24% | - | Total fail |
| MSE loss training    | 0.6%  | - | Wrong loss type |
| CE loss (old layers) | 6.9%  | - | Better, still not great |
| CE loss (new layers) | 18.6% | 1.82x | Before MASK fix |
| + MASK embedding fix | **24.7%** | **1.75x** | **Current best** |

During training: val_t1=90.4%, val_prefix=55.0%

### Latest Evaluation (2026-01-13)

**Training config:**
- Draft model: 5-layer Qwen3 (from `/models/Qwen3-8B-DFlash-b16`)
- Target layers: `[24, 30, 31, 32, 34]`
- Training data: chunks 0-255 `stride=1`
- Best val_loss: 1.4457

**Held-out evaluation (chunks 400-401, 10 samples, after MASK fix):**
| Metric | Value |
|--------|-------|
| Avg acceptance rate | **24.7%** |
| Avg acceptance length | **2.73** |
| Avg speedup | 1.75x |

## Layer Selection

1. **Step A**: Cosine similarity to final layer - found candidates with high similarity
2. **Step B**: Early-exit CE loss - picked layers with lowest loss

**New layers: `[24, 30, 31, 32, 34]`** - all in upper half of model, makes sense.

## Model Structure (for reference)

```
AlpamayoR1
└── vlm (Qwen3VLForConditionalGeneration)
    └── model.language_model
        ├── embed_tokens
        ├── layers[0-35]
        └── norm
    └── lm_head
```

Access paths:
- `vlm.model.language_model.norm` - final norm
- `vlm.lm_head` - language model head
- `vlm.model.language_model.embed_tokens` - embeddings

## Key Files

- `src/alpamayo_r1/dflash_integration.py` - main accelerator class
- `dflash/generate_distillation_data.py` - data generation
- `dflash/train_dflash.py` - **offline training (CE/KL loss, prefix-weighted)**

## Commands
### Step 1: Choose layers
```bash
cd dflash

# Step 1: Cosine similarity (8 GPUs)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python compute_cosine_similarity.py \
        --rank $i --world-size 8 --num-samples 32 &
done
wait
python compute_cosine_similarity.py --aggregate --plot

# Step 2: Early-exit loss for candidate layers (8 GPUs)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python compute_early_exit_loss.py \
        --rank $i --world-size 8 --num-samples 32 \
        --layers "0,16,20,24,28,30,31,32,34" &
done
wait
python compute_early_exit_loss.py --aggregate --plot --layers "0,16,20,24,28,30,31,32,34"
```

### Step 2: Generate distillation data (400 chunks, 8 GPUs) - for offline training
```bash
./run_distillation_data.sh
```

### Step 3: Offline training (prefix-weighted)
```bash
torchrun --nproc_per_node=8 dflash/train_dflash.py \
      --data-dir /data/dflash_distillation \
      --loss-type ce+kl \
      --full-vocab \
      --learning-rate 1e-5
```

### Step 4: Evaluate on held-out data
Debug (verbose)
```bash
python src/alpamayo_r1/test_dflash_inference.py \
      --draft-model /exp/dflash_0114_200248/best \
      --clip-ids /data/physicalai_av/clip_ids_400_401.json \
      --num-samples 10 \
      --max-tokens 64 \
      --verbose
```

Compare
```bash
python src/alpamayo_r1/test_dflash_inference.py \
      --draft-model /exp/dflash_0114_200248/best \
      --clip-ids /data/physicalai_av/clip_ids_400_401.json \
      --num-samples 10 \
      --max-tokens 64 \
      --compare
```

## Issues

### Training-Inference Mismatch (2026-01-14)

**Root cause of low acceptance rate identified via fingerprint analysis.**

The training data contains **only mid-CoC text tokens**. It completely lacks:

| Token Type | Training Data | Inference Needs |
|------------|---------------|-----------------|
| `<\|cot_start\|>` (155677) | 5,377 samples | First token of generation |
| `<\|cot_end\|>` (155678) | **0 samples** | End of every CoC |
| `<\|traj_future_start\|>` (155681) | **0 samples** | After every CoC |
| Trajectory tokens (151669-155668) | **0 samples** | 81% of generated tokens |

**Observed per-block acceptance pattern:**

```
Block  1 (BOUNDARY): accepted 1 token   ← boundary position issue
Block  2 (MID-COC):  accepted 8 tokens  ← perfect!
Block  3 (MID-COC):  accepted 8 tokens  ← perfect!
Block  4 (COC-END):  accepted 1 token   ← collapse starts here
Block  5+:           accepted 1 token   ← never recovers
```

**Why this happens:**
1. Blocks 2-3 predict mid-CoC text → matches training data → high accuracy
2. Block 4 must predict `<|cot_end|>` → never in training → 0% accuracy
3. All subsequent blocks fail because draft never learned these token transitions

**Additional finding - Boundary vs Non-Boundary accuracy:**

| Sample Type | First-Token Accuracy | Position-1 Accuracy |
|-------------|---------------------|---------------------|
| Boundary (7.6% of training) | 61.8% | 68.5% |
| Non-boundary (92.4% of training) | 88.4% | 90.6% |

The `target_hidden` distribution at boundary positions differs significantly from mid-sequence (mean norm 881 vs 979, 96% of dimensions differ by >0.1).

**Solutions:**
1. Include `<|cot_end|>` and trajectory token transitions in training data
2. OR stop DFlash decoding before `<|cot_end|>` and fall back to standard decoding
3. Balance boundary vs non-boundary samples in training (currently 7.6% vs 92.4%)