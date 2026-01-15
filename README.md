# DFlash + Alpamayo Integration

Notes on integrating DFlash speculative decoding with Alpamayo CoC generation.

## Goal

Trying to speed up Alpamayo's Chain-of-Causation text generation using DFlash block diffusion. Only targeting the text part, not action decoding.

## Progress

**Distillation training complete. Evaluating acceptance rates.**

## Results

During training: val_t1=90.4%, val_prefix=55.0%

### Latest Evaluation 1/14

**Training config:**
- Draft model: 5-layer Qwen3 (from `/models/Qwen3-8B-DFlash-b16`)
- Target layers: `[24, 30, 31, 32, 34]`
- Training data: chunks 0-399 `stride=1`

**Held-out evaluation (chunks 400-401, 10 samples, after MASK fix):**
| Metric | Value |
|--------|-------|
| Avg acceptance rate | **39.2%** |
| Avg acceptance length | **3.74** |
| Avg speedup | 1.77x |

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

### Training-Inference Mismatch 1/14

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

**Critical finding - Repetitive token degeneration (2026-01-14):**

When the drafter reaches the end of CoC where it should generate `<|cot_end|>`, it degenerates into repetitive high-frequency tokens:

```
Draft decoded: [' straight', ' ahead', ' the', ' the', ' the', ' the', ' is', ' the']
Draft decoded: [' traffic', ' in', ' the', ' the', 'about', ' ahead', ' the', ' the']
Draft decoded: [' green', ' light', ' is', ' the', ' the', ' the', ' intersection', ' the']
```

This happens because:
- The model has never seen `<|cot_end|>` (token 155678) during training
- When it should stop, it falls back to common tokens like "the"
- The stop-token mechanism fails because `<|cot_end|>` is never generated
- This creates infinite loops of garbage tokens

**Root cause in data generation (`generate_distillation_data.py`):**
```python
# Line 116-117: Only extracts TEXT content, excludes special tokens!
cot_text = extra["cot"][0][0][0]
cot_tokens = model.tokenizer(cot_text, add_special_tokens=False, ...)
```

**Additional finding - Boundary vs Non-Boundary accuracy:**

| Sample Type | First-Token Accuracy | Position-1 Accuracy |
|-------------|---------------------|---------------------|
| Boundary (7.6% of training) | 61.8% | 68.5% |
| Non-boundary (92.4% of training) | 88.4% | 90.6% |

The `target_hidden` distribution at boundary positions differs significantly from mid-sequence (mean norm 881 vs 979, 96% of dimensions differ by >0.1).

**Solutions:**
1. **Regenerate training data with `<|cot_end|>` included** ← Required fix
   - Modify `generate_distillation_data.py` to use full token sequence instead of just text content
   - Include the transition: `"... is red." → <|cot_end|>`
   - Retrain DFlash so it learns when to stop
2. ~~Stop DFlash decoding before `<|cot_end|>` and fall back to standard decoding~~ ❌ (doesn't work - drafter never generates `<|cot_end|>`)
3. Balance boundary vs non-boundary samples in training (currently 7.6% vs 92.4%)

### Hidden State Mismatch Bug 1/14 - FIXED

**Critical bug causing mode collapse after rejection.**

**Symptom:** After a rejection in Step N, Step N+1 produces gibberish like `' the the the straight the is the'`.

**Example from verbose log:**
```
>>> STEP 1 (position 3006) - PERFECT
  Draft:  'Accelerate to proceed through the intersection since'
  Target: 'ate to proceed through the intersection since the'
  Accepted: 8 tokens (ALL MATCH) ✓

>>> STEP 2 (position 3014) - REJECTION
  Draft:  '... since the RIGHT-turn ...'  ← DFlash guessed wrong direction
  Target: '... since the STRAIGHT ...'    ← Target corrected it
  Accepted: 1 token ('the') + correction ('straight')

>>> STEP 3 (position 3015) - MODE COLLAPSE
  Draft:  'straight ahead the the straight the is the'  ← GIBBERISH!
  Target: 'traffic is turning left traffic turn clear'
  Accepted: 1 token
```

**Root Cause:**

When a rejection occurs at position K:
1. The `verify_output` hidden states were computed with the WRONG draft token ("RIGHT")
2. But the correction token is "STRAIGHT"
3. We extract hidden state at position K from `verify_output` - this hidden state corresponds to "RIGHT"
4. We feed this hidden state to DFlash and tell it the context is "STRAIGHT"
5. **Mismatch**: DFlash receives hidden state for "RIGHT" but is told it's "STRAIGHT" → confusion → mode collapse

**The Fix: Correction Pass**

When a rejection occurs (`acceptance_length < block_size - 1`), we must run the target model once more on the correction token to get its TRUE hidden state:

```python
if acceptance_length == block_size - 1:
    # Full match: use hidden from verify_output (correct)
    target_hidden = extract_hidden(verify_output, -1)
else:
    # Rejection: hidden at that position is WRONG (computed under draft token)
    # Must run "correction pass" to get true hidden for correction token
    correction_output = target_vlm(
        input_ids=correction_token_id,  # The actual accepted token
        position_ids=correction_pos,
        past_key_values=kv_cache,       # Reuse cache
        output_hidden_states=True,
    )
    target_hidden = extract_hidden(correction_output, -1)
```

**Key Insight:** Step 1's perfect acceptance proves the model training is correct. The bug was purely in the inference logic - reusing hidden states computed under wrong tokens.