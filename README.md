# DFlash + Alpamayo Integration

Notes on integrating DFlash speculative decoding with Alpamayo CoC generation.

## Goal

Trying to speed up Alpamayo's Chain-of-Causation text generation using DFlash block diffusion. Only targeting the text part, not action decoding.

## Results

### Latest Evaluation 1/15

**Training config:**
- Draft model: 5-layer Qwen3-based (~2.3B params, ~1B trainable)
- Target layers: `[24, 30, 31, 32, 34]`
- Training data: chunks 0-399 `stride=1`

**Held-out evaluation (chunks 400-401, 10 clips):**
| Metric | Value |
|--------|-------|
| Avg acceptance rate | **75.8%** |
| Avg acceptance length | **6.25** |
| Avg speedup | 2.18x (952.3 ms vs 2066.4 ms)|

## Layer Selection

1. **Step A**: Cosine similarity to final layer - found candidates with high similarity
2. **Step B**: Early-exit CE loss - picked layers with lowest loss

**New layers: `[24, 30, 31, 32, 34]`** - all in upper half of model, makes sense.

## Model Structure

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

```
├── src/alpamayo_r1/
│   ├── dflash_integration.py      # Main accelerator class
│   └── test_dflash_inference.py   # Evaluation (--verbose, --compare)
│
└── dflash/
    ├── train_dflash.py            # Train from scratch (recommended)
    ├── distill_dflash.py          # Distillation from pre-trained
    ├── generate_distillation_data.py
    └── run_training.sh            # Launch script
```

## Instructions
### Step 1: Choose layers
```bash
cd dflash

# Step 1.1: Cosine similarity (8 GPUs)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python compute_cosine_similarity.py \
        --rank $i --world-size 8 --num-samples 32 &
done
wait
python compute_cosine_similarity.py --aggregate --plot

# Step 1.2: Early-exit loss for candidate layers (8 GPUs)
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

### Step 3: Train DFlash (recommended)
Creates a new DFlash model with Alpamayo's vocabulary (155,698 tokens including trajectory tokens).
```bash
cd dflash
./run_training.sh
```

Or manually:
```bash
torchrun --nproc_per_node=8 dflash/train_dflash.py \
    --target-model /models/Alpamayo-R1-10B \
    --data-dir /data/dflash_distillation \
    --output-dir /exp \
    --num-epochs 5 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --num-draft-layers 5
```

Key features:
- Uses Alpamayo's vocab_size (155,697 + 1 MASK token)
- Loads `embed_tokens` and `lm_head` from Alpamayo VLM
- Supports `--train-embeddings` flag to make embeddings trainable

### Step 3 (alt): Distillation from pre-trained DFlash
```bash
torchrun --nproc_per_node=8 dflash/distill_dflash.py \
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
      --temperature 0 \
      --verbose
```

Compare
```bash
python src/alpamayo_r1/test_dflash_inference.py \
      --draft-model /exp/dflash_0114_200248/best \
      --clip-ids /data/physicalai_av/clip_ids_400_401.json \
      --num-samples 10 \
      --max-tokens 64 \
      --temperature 0 \
      --compare
```

## Issues

### Inference Heuristics

The draft model has difficulty with sentence boundaries. These heuristics improve end-of-generation handling:

**1. Period → End Token Shortcut**
When `first_in_block` is `.` (token 13), directly append `<|cot_end|>` and stop generation.
```
Before: "." → draft model → garbage tokens → miss stop
After:  "." → directly append <|cot_end|> → done
```
This skips the draft model entirely since we know the next token should be `<|cot_end|>`.

**2. Premature End Token Replacement**
If draft predicts `<|cot_end|>` without a preceding `.`, replace it with `.`
```
Draft:  [..., "ahead", <|cot_end|>, ...]  ← missing period
Fixed:  [..., "ahead", ".", ...]          ← replaced, next step triggers heuristic 1
```

**3. Mid-Block Stop Detection**
Check for stop tokens anywhere in the accepted sequence, not just at the end. Truncate and stop immediately when found.

These are implemented in both `dflash_integration.py` and `test_dflash_inference.py`.
