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
| CE loss (new layers) | **18.6%** | **1.82x** | - |

### Latest Evaluation (2026-01-13)

**Training config:**
- Draft model: 5-layer Qwen3 (from `/models/Qwen3-8B-DFlash-b16`)
- Target layers: `[24, 30, 31, 32, 34]`
- Training data: chunks 0-255 `stride=1`
- Best val_loss: 1.4457

**Held-out evaluation (chunks 400-401, 10 samples):**
| Metric | Value |
|--------|-------|
| Avg acceptance rate | 18.6% |
| Avg acceptance length | 2.31 |
| Avg speedup | 1.82x |

Individual variance is high (5% to 50%) due to scenario complexity, not memorization.

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
- `dflash/compute_cosine_similarity.py` - layer similarity analysis
- `dflash/compute_early_exit_loss.py` - early-exit loss analysis
- `dflash/generate_distillation_data.py` - data gen
- `dflash/train_dflash.py` - training

## Commands

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

# Step 3: Generate distillation data (400 chunks, 8 GPUs)
./run_distillation_data.sh

# Step 4: Train
torchrun --nproc_per_node=8 train_dflash.py --data-dir /data/dflash_distillation
```

## Issues

**Vocabulary mismatch (Critical):**
- Draft model vocab: 151,936 tokens (standard Qwen3)
- Target model vocab: 155,697 tokens (extended with trajectory tokens)
- ~4,000 trajectory tokens cannot be predicted by draft model
- ~7% of tokens in training data are masked due to this

**High validation loss:**
- Best val_loss = 1.4457 in 5 epochs

**Batch size**: `1` for speculative decoding (different seqs have different acceptance lengths).

## Next Steps (Priority Order)

1. **Extend draft vocabulary** - Train draft model with full 155,697 vocab including trajectory tokens
2. **Add KL divergence loss** - Combine CE + KL (0.5 each) for better distribution matching
3. **Increase draft capacity** - Try 8-12 layers instead of 5
4. **More training data** - 294k blocks may be insufficient for this complex domain
