#!/usr/bin/env python3
"""DFlash integration for accelerating Alpamayo Chain-of-Causation reasoning generation.

This module provides utilities to use DFlash block diffusion speculative decoding
to accelerate the autoregressive text generation in Alpamayo's VLM backbone.

Usage:
    from alpamayo_r1.dflash_integration import DFlashAlpamayoAccelerator, load_dflash_draft_model

    # Load models
    alpamayo = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", ...)
    draft_model = load_dflash_draft_model("z-lab/Qwen3-8B-DFlash-b16")

    # Create accelerator
    accelerator = DFlashAlpamayoAccelerator(draft_model, alpamayo.vlm, alpamayo.tokenizer)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import DynamicCache
from transformers.generation.logits_process import LogitsProcessor

logger = logging.getLogger(__name__)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Build target layer IDs for context feature extraction.

    Samples layer indices evenly distributed across the target model's layers.
    This allows the draft model to condition on multi-scale representations.

    Args:
        num_target_layers: Number of layers in the target model.
        num_draft_layers: Number of layers in the draft model.

    Returns:
        List of layer indices to extract features from.
    """
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """Extract and concatenate hidden states from specified layers.

    Args:
        hidden_states: Tuple of hidden states from all layers (including embeddings).
        layer_ids: List of layer indices to extract from.

    Returns:
        Concatenated hidden states tensor of shape (batch, seq_len, hidden_size * num_layers).
    """
    offset = 1  # Account for embedding layer in hidden_states
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Sample tokens from logits with optional temperature and logits processing.

    Args:
        logits: Logits tensor of shape (batch, seq_len, vocab_size).
        temperature: Sampling temperature. 0 means greedy decoding.
        logits_processor: Optional processor to apply before sampling.

    Returns:
        Sampled token IDs of shape (batch, seq_len).
    """
    bsz, seq_len, vocab_size = logits.shape

    # Apply logits processor if provided (e.g., to mask trajectory tokens)
    if logits_processor is not None:
        # Process each position independently (processor expects 2D input)
        processed_logits = []
        for pos in range(seq_len):
            pos_logits = logits_processor(None, logits[:, pos, :])
            processed_logits.append(pos_logits)
        logits = torch.stack(processed_logits, dim=1)

    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


class _TrajectoryTokenMask(LogitsProcessor):
    """Masks out logits for discrete trajectory tokens during CoC generation.

    This prevents the model from generating action/trajectory tokens prematurely
    during Chain-of-Causation text generation.
    """

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(
        self, input_ids: torch.LongTensor | None, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


def setup_tokenizer_for_dflash(
    tokenizer,
    model,
    mask_token_id_override: int | None = None,
    init_to_mean: bool = True,
) -> int:
    """Ensure tokenizer has a MASK token and model embeddings are resized.

    DFlash uses MASK tokens for block diffusion. This function sets up the mask token,
    preferring existing tokens or overrides to avoid out-of-distribution embeddings.

    IMPORTANT: For pretrained DFlash checkpoints, the mask token must match what was
    used during training. Using a mismatched token causes severe quality degradation.

    Args:
        tokenizer: The tokenizer to modify.
        model: The model whose embeddings may need resizing.
        mask_token_id_override: If set, use this token ID directly (for pretrained DFlash).
        init_to_mean: If adding a new token, initialize its embedding to mean of vocab.

    Returns:
        The MASK token ID.
    """
    # Option 1: Use override
    if mask_token_id_override is not None:
        vocab_size = model.get_input_embeddings().weight.shape[0]
        if mask_token_id_override >= vocab_size:
            raise ValueError(
                f"mask_token_id_override={mask_token_id_override} out of vocab range ({vocab_size})"
            )
        logger.info(f"[DFlash Setup] Using override mask_token_id={mask_token_id_override}")
        return mask_token_id_override

    # Option 2: Use existing mask token
    if tokenizer.mask_token is not None:
        logger.info(
            f"[DFlash Setup] Mask token exists: {tokenizer.mask_token} "
            f"(ID: {tokenizer.mask_token_id})"
        )
        return tokenizer.mask_token_id

    # Option 3: Try to find reserved token
    for reserved_token in ["<|extra_0|>", "<|placeholder|>", "[MASK]"]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(reserved_token)
            if token_id != tokenizer.unk_token_id:
                logger.info(f"[DFlash Setup] Using existing token '{reserved_token}' (ID: {token_id})")
                tokenizer.mask_token = reserved_token
                return token_id
        except Exception:
            continue

    # Option 4: Add new token (last resort)
    logger.warning(
        "[DFlash Setup] Adding new <|MASK|> token. "
        "WARNING: May cause poor draft quality with pretrained DFlash checkpoints."
    )
    tokenizer.add_special_tokens({'mask_token': '<|MASK|>'})
    logger.info(f"[DFlash Setup] Added <|MASK|> token. ID: {tokenizer.mask_token_id}")

    # Resize embeddings
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > current_vocab_size:
        logger.info(
            f"[DFlash Setup] Resizing model embeddings from {current_vocab_size} "
            f"to {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new embedding to mean for stability
        if init_to_mean:
            input_embeddings = model.get_input_embeddings()
            mask_id = tokenizer.mask_token_id
            with torch.no_grad():
                mean_embedding = input_embeddings.weight[:-1].mean(dim=0)
                input_embeddings.weight[mask_id] = mean_embedding
            logger.info("[DFlash Setup] Initialized MASK embedding to mean of vocabulary.")

    return tokenizer.mask_token_id


@dataclass
class DFlashConfig:
    """Configuration for DFlash speculative decoding."""

    block_size: int = 8
    mask_token: str = "<|MASK|>"
    temperature: float = 0.0
    # Layer IDs to extract from target model (auto-computed if None)
    target_layer_ids: list[int] | None = None
    # Trajectory token masking (required for Alpamayo CoC generation)
    traj_token_offset: int | None = None
    traj_vocab_size: int | None = None
    # If set, use this existing token ID as the mask token instead of adding a new one.
    # This is critical for pretrained DFlash checkpoints that were trained with a specific
    # mask token (e.g., <|extra_0|> in Qwen vocabulary). Using a mismatched mask token
    # will cause severe degradation in draft quality.
    mask_token_id_override: int | None = None
    # If True and a new mask token must be added, initialize its embedding to the mean
    # of all existing embeddings rather than random. This provides more stable behavior
    # when the exact mask token from DFlash training is unknown.
    init_mask_to_mean: bool = True


@dataclass
class GenerationStats:
    """Statistics from speculative generation."""

    total_tokens: int = 0
    total_iterations: int = 0
    acceptance_lengths: list[int] = field(default_factory=list)
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    block_size: int = 8  # Must match accelerator's block_size

    @property
    def mean_acceptance_length(self) -> float:
        """Mean number of tokens accepted per iteration."""
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        if not self.acceptance_lengths:
            return 0.0
        # Each iteration drafts block_size-1 tokens, accepts acceptance_length-1 + 1 from target
        total_drafted = len(self.acceptance_lengths) * (self.block_size - 1)
        total_accepted = sum(max(0, a - 1) for a in self.acceptance_lengths)
        return total_accepted / total_drafted if total_drafted > 0 else 0.0


class DFlashAlpamayoAccelerator:
    """Accelerator for Alpamayo CoC generation using DFlash speculative decoding.

    This class wraps a DFlash draft model and provides methods to accelerate
    the Chain-of-Causation text generation in Alpamayo's inference pipeline.

    The accelerator performs:
    1. Multimodal prefill with visual inputs (handled by target VLM)
    2. Speculative decoding loop for text generation (DFlash acceleration)

    Example:
        accelerator = DFlashAlpamayoAccelerator(draft_model, vlm, tokenizer)
        output_ids, stats = accelerator.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=256,
        )
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_vlm: nn.Module,
        tokenizer: Any,
        config: DFlashConfig | None = None,
    ):
        """Initialize the accelerator.

        Args:
            draft_model: The DFlash draft model (DFlashDraftModel).
            target_vlm: The target VLM (Qwen3VLForConditionalGeneration).
            tokenizer: The tokenizer with mask token support.
            config: Configuration for speculative decoding.
        """
        self.draft_model = draft_model
        self.target_vlm = target_vlm
        self.tokenizer = tokenizer
        self.config = config or DFlashConfig()

        # Get block size from draft model or config
        self.block_size = getattr(draft_model, 'block_size', self.config.block_size)

        # Get or compute target layer IDs
        if self.config.target_layer_ids is not None:
            self.target_layer_ids = self.config.target_layer_ids
        elif hasattr(draft_model, 'target_layer_ids'):
            self.target_layer_ids = draft_model.target_layer_ids
        else:
            # Compute based on model configs
            num_target_layers = target_vlm.config.text_config.num_hidden_layers
            num_draft_layers = draft_model.config.num_hidden_layers
            self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

        # Setup mask token
        self.mask_token_id = self._setup_mask_token()

        # Cache for embed_tokens and lm_head references
        # Qwen3VL structure: vlm.model.language_model.embed_tokens
        self._embed_tokens = target_vlm.model.language_model.embed_tokens
        self._lm_head = target_vlm.lm_head

        # Setup logits processor for trajectory token masking (prevents premature action tokens)
        self._logits_processor = None
        if self.config.traj_token_offset is not None and self.config.traj_vocab_size is not None:
            self._logits_processor = _TrajectoryTokenMask(
                traj_token_offset=self.config.traj_token_offset,
                traj_vocab_size=self.config.traj_vocab_size,
            )
            logger.info(
                f"[DFlash] Trajectory token masking enabled: "
                f"offset={self.config.traj_token_offset}, vocab_size={self.config.traj_vocab_size}"
            )

        logger.info(
            f"DFlash accelerator initialized: block_size={self.block_size}, "
            f"target_layers={self.target_layer_ids}, mask_token_id={self.mask_token_id}"
        )

    def _setup_mask_token(self) -> int:
        """Setup the tokenizer with mask token for DFlash.

        For pretrained DFlash checkpoints, it's critical to use the same mask token
        that was used during training. Using a mismatched mask token will cause
        the draft model to receive out-of-distribution inputs, severely degrading quality.

        Returns:
            The mask token ID.
        """
        # Option 1: Use override if specified (preferred for pretrained DFlash)
        if self.config.mask_token_id_override is not None:
            mask_id = self.config.mask_token_id_override
            vocab_size = self.target_vlm.get_input_embeddings().weight.shape[0]
            # If mask_id is at vocab boundary, resize embeddings (training added this token)
            if mask_id == vocab_size:
                logger.info(f"[DFlash] Resizing embeddings from {vocab_size} to {vocab_size + 1} for mask token")
                self.target_vlm.resize_token_embeddings(vocab_size + 1)
            elif mask_id > vocab_size:
                raise ValueError(
                    f"mask_token_id_override={mask_id} is out of vocabulary range ({vocab_size}). "
                    "Ensure the ID matches the token used during DFlash training."
                )
            # CRITICAL: Always reset mask embedding to mean to match training distribution
            if self.config.init_mask_to_mean:
                logger.info(f"[DFlash] Resetting mask token {mask_id} embedding to vocab mean (matching training)")
                self._init_mask_embedding_to_mean(mask_id)
            logger.info(f"[DFlash] Using override mask_token_id={mask_id}")
            return mask_id

        # Option 2: Check if tokenizer already has a mask token
        if self.tokenizer.mask_token_id is not None:
            logger.info(
                f"[DFlash] Using existing mask token: {self.tokenizer.mask_token} "
                f"(ID: {self.tokenizer.mask_token_id})"
            )
            # CRITICAL: Reset existing mask embedding to mean to match training distribution
            if self.config.init_mask_to_mean:
                logger.info(f"[DFlash] Resetting mask token {self.tokenizer.mask_token_id} embedding to vocab mean (matching training)")
                self._init_mask_embedding_to_mean(self.tokenizer.mask_token_id)
            return self.tokenizer.mask_token_id

        # Option 3: Try to find a reserved token that DFlash might have used
        # Many DFlash implementations use <|extra_0|> or similar reserved tokens
        for reserved_token in ["<|extra_0|>", "<|placeholder|>", "[MASK]"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(reserved_token)
                if token_id != self.tokenizer.unk_token_id:
                    logger.info(
                        f"[DFlash] Found potential mask token '{reserved_token}' (ID: {token_id}). "
                        "If draft quality is poor, verify this matches the DFlash training config."
                    )
                    self.tokenizer.mask_token = reserved_token
                    # CRITICAL: Reset reserved token embedding to mean to match training distribution
                    if self.config.init_mask_to_mean:
                        logger.info(f"[DFlash] Resetting reserved token {token_id} embedding to vocab mean (matching training)")
                        self._init_mask_embedding_to_mean(token_id)
                    return token_id
            except Exception:
                continue

        # Option 4: Add new token as last resort
        logger.warning(
            "[DFlash] No existing mask token found. Adding new token. "
            "WARNING: If using a pretrained DFlash checkpoint, this may cause poor draft quality. "
            "Set mask_token_id_override in DFlashConfig to the correct token ID."
        )
        num_added = self.tokenizer.add_special_tokens({"mask_token": self.config.mask_token})
        if num_added > 0:
            # Resize embeddings
            self.target_vlm.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"[DFlash] Added mask token '{self.config.mask_token}' to tokenizer")

            # Initialize new embedding to mean of existing embeddings for stability
            if self.config.init_mask_to_mean:
                self._init_mask_embedding_to_mean()

        return self.tokenizer.mask_token_id

    def _init_mask_embedding_to_mean(self, mask_id: int | None = None) -> None:
        """Initialize the mask token embedding to the mean of all existing embeddings.

        This provides more stable behavior than random initialization when the exact
        mask token from DFlash training is unknown.
        """
        input_embeddings = self.target_vlm.get_input_embeddings()
        if mask_id is None:
            mask_id = self.tokenizer.mask_token_id

        with torch.no_grad():
            # Compute mean of all embeddings (excluding the new mask token)
            all_embeddings = input_embeddings.weight[:-1]  # Exclude last (new) token
            mean_embedding = all_embeddings.mean(dim=0)
            input_embeddings.weight[mask_id] = mean_embedding

        logger.info(
            f"[DFlash] Initialized mask token embedding (ID: {mask_id}) to mean of vocabulary. "
            "For best results, use the exact mask token from DFlash training."
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        stop_token_ids: list[int] | None = None,
        temperature: float | None = None,
        position_ids: torch.Tensor | None = None,
        rope_deltas: torch.Tensor | None = None,
        **prefill_kwargs,
    ) -> tuple[torch.Tensor, GenerationStats]:
        """Generate text using speculative decoding with DFlash.

        This method performs:
        1. Multimodal prefill with visual inputs
        2. Speculative decoding loop for text generation

        Args:
            input_ids: Input token IDs including prompt, shape (1, seq_len).
            pixel_values: Visual inputs (images), optional.
            image_grid_thw: Image grid dimensions, optional.
            max_new_tokens: Maximum new tokens to generate.
            stop_token_ids: Token IDs that stop generation.
            temperature: Sampling temperature (overrides config).
            position_ids: Optional 3D position IDs for MROPE (shape: 3, batch, seq_len).
                If not provided, will be computed during prefill.
            rope_deltas: Optional rope deltas from previous processing.
            **prefill_kwargs: Additional kwargs for prefill.

        Returns:
            Tuple of (output_ids, generation_stats).
        """
        self.draft_model.eval()
        self.target_vlm.eval()

        device = input_ids.device
        bsz = input_ids.shape[0]
        temperature = temperature if temperature is not None else self.config.temperature
        block_size = self.block_size
        stats = GenerationStats(block_size=block_size)

        # Speculative decoding with token-level acceptance requires batch_size=1
        # (different sequences would have different acceptance lengths)
        if bsz != 1:
            raise ValueError(
                f"DFlash speculative decoding requires batch_size=1, got {bsz}. "
                "For batched inference, process sequences sequentially or use standard generation."
            )

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        # Initialize output buffer with mask tokens
        output_ids = torch.full(
            (bsz, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Initialize KV caches
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # ====== PREFILL STAGE ======
        import time
        prefill_start = time.perf_counter()

        # Process multimodal inputs (images + text prompt)
        # NOTE: For Qwen2-VL/Qwen3-VL with MROPE, we let the model compute position_ids internally
        # during prefill when pixel_values are provided. The model's internal logic handles
        # the 3D positional encoding for image tokens correctly.
        prefill_kwargs_internal = dict(prefill_kwargs)
        if pixel_values is not None:
            # Let the VLM compute MROPE position_ids internally for multimodal inputs
            prefill_output = self.target_vlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
                **prefill_kwargs_internal,
            )
            # Capture rope_deltas for continuation (Qwen2-VL/Qwen3-VL stores this)
            rope_deltas = getattr(self.target_vlm.model, "rope_deltas", None)
        else:
            # Text-only: use simple position IDs
            simple_position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
            prefill_output = self.target_vlm(
                input_ids=input_ids,
                position_ids=simple_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
                **prefill_kwargs_internal,
            )
            rope_deltas = None

        # Copy input tokens to output
        output_ids[:, :num_input_tokens] = input_ids

        # Sample first token from prefill logits (apply logits processor if configured)
        first_token_logits = prefill_output.logits[:, -1:, :]
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_tokens(
            first_token_logits, temperature, self._logits_processor
        )

        # Extract context features for draft model
        # In stateless mode, we only keep the LAST hidden state from prefill.
        # This hidden state has already attended to all image and prompt features
        # through Alpamayo's self-attention, so it contains compressed context.
        full_hidden = extract_context_feature(
            prefill_output.hidden_states, self.target_layer_ids
        )
        # Keep only the last position (the first generated token's context)
        target_hidden = full_hidden[:, -1:, :]  # (B, 1, hidden_dim)

        # Track the current sequence length for position ID continuation
        # After prefill, the KV cache contains num_input_tokens entries
        current_seq_len = num_input_tokens

        stats.prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

        # ====== DECODE STAGE ======
        decode_start = time.perf_counter()
        start = num_input_tokens

        while start < max_length:
            # 1. Prepare the block (first token is known, rest are masks)
            block_output_ids = output_ids[:, start : start + block_size].clone()

            # Compute position IDs for TARGET MODEL verification (uses absolute positions)
            # For MROPE (Qwen2-VL/Qwen3-VL), position_ids should be 3D: (3, batch, seq_len)
            block_positions = torch.arange(
                start, start + block_size, device=device, dtype=torch.long
            )
            if rope_deltas is not None:
                # MROPE: Create 3D position IDs, applying rope_deltas offset
                block_position_ids = block_positions.view(1, 1, -1).expand(3, bsz, -1).clone()
                block_position_ids = block_position_ids + rope_deltas.unsqueeze(-1).to(
                    dtype=torch.long, device=device
                )
            else:
                # Standard 1D position IDs
                block_position_ids = block_positions.unsqueeze(0).expand(bsz, -1)

            # 2. Position reset for DRAFT model (Relative Positioning)
            # During training, DFlash always sees positions 0..ctx_len+block_size. Inference must strictly match this behavior to avoid distribution shift. ctx_len=1 (single hidden state context) + block_size tokens
            reset_position_ids = torch.arange(
                0, 1 + block_size, device=device
            ).unsqueeze(0).expand(bsz, -1)

            # 3. Embedding
            noise_embedding = self._embed_tokens(block_output_ids)

            # 4. Context slicing (Stateless Mode)
            # During training, DFlash only observes a single hidden vector (ctx_len=1). We take only the last step of target_hidden - this contains a compressed representation that has already attended to all prior context (including images) through Alpamayo's self-attention layers.
            current_context = target_hidden[:, -1:, :]  # (B, 1, hidden_dim)

            # 5. [CRITICAL] Disable KV cache
            # Since we operate in stateless mode with position reset, past_key_values are unnecessary and would cause incorrect behavior.
            draft_hidden = self.draft_model(
                target_hidden=current_context,      # (B, 1, H)
                noise_embedding=noise_embedding,    # (B, block_size, H)
                position_ids=reset_position_ids,    # (B, 1+block_size), always starting from 0
                past_key_values=None,               # Disable cache
                use_cache=False,                    # Do not return cache
                is_causal=False,
            )

            # Get logits for drafted tokens (excluding first known token)
            # draft_hidden shape: (B, block_size, hidden_dim)
            draft_logits = self._lm_head(draft_hidden[:, 1:, :])

            # Sample draft tokens (apply logits processor to prevent trajectory tokens)
            block_output_ids[:, 1:] = sample_tokens(
                draft_logits, temperature, self._logits_processor
            )

            # Verify: Run target model on the drafted block
            verify_output = self.target_vlm(
                input_ids=block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            # Sample from target model's logits (apply logits processor)
            posterior = sample_tokens(
                verify_output.logits, temperature, self._logits_processor
            )

            # Compute acceptance: count consecutive matching tokens
            matches = block_output_ids[:, 1:] == posterior[:, :-1]
            acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

            # Accept matched tokens + target's next token
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            # Check for stop tokens BEFORE updating position
            # The last accepted token is posterior[0, acceptance_length]
            last_accepted_token = posterior[0, acceptance_length].item()
            hit_stop = stop_token_ids is not None and last_accepted_token in stop_token_ids

            # Update position and stats
            start += acceptance_length + 1
            stats.acceptance_lengths.append(acceptance_length + 1)
            stats.total_iterations += 1

            # Stop immediately if we hit stop token
            if hit_stop:
                break

            # Update caches - crop to the new valid length
            past_key_values_target.crop(start)

            # Extract hidden state for next iteration's context.
            # CRITICAL: When a rejection occurs, the hidden state at position acceptance_length
            # in verify_output was computed under the WRONG draft token, not the correction token.
            # We must run a "correction pass" to get the true hidden state.

            if acceptance_length == block_size - 1:
                # Full match (all drafted tokens accepted): hidden state is correct
                new_hidden = extract_context_feature(
                    verify_output.hidden_states, self.target_layer_ids
                )
                target_hidden = new_hidden[:, -1:, :]
            else:
                # Rejection occurred: hidden state at acceptance_length is WRONG
                # (computed under draft token, but we need hidden for correction token)
                # Run correction pass on the actual accepted token

                # The correction token is at position (start - 1) in output_ids
                correction_token_id = output_ids[:, start - 1 : start]  # (B, 1)

                # Prepare position IDs for the correction token
                correction_pos = start - 1
                if rope_deltas is not None:
                    # MROPE: Create 3D position IDs
                    corr_pos_tensor = torch.tensor([correction_pos], device=device, dtype=torch.long)
                    corr_position_ids = corr_pos_tensor.view(1, 1, 1).expand(3, bsz, 1).clone()
                    corr_position_ids = corr_position_ids + rope_deltas.unsqueeze(-1).to(
                        dtype=torch.long, device=device
                    )
                else:
                    corr_position_ids = torch.tensor([[correction_pos]], device=device, dtype=torch.long)

                # Run correction pass to get true hidden state
                correction_output = self.target_vlm(
                    input_ids=correction_token_id,
                    position_ids=corr_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )

                # Extract hidden state from correction pass
                new_hidden = extract_context_feature(
                    correction_output.hidden_states, self.target_layer_ids
                )
                target_hidden = new_hidden[:, -1:, :]

        stats.decode_time_ms = (time.perf_counter() - decode_start) * 1000

        # Cleanup output
        output_ids = output_ids[:, :max_length]
        # Remove mask tokens
        mask = output_ids[0] != self.mask_token_id
        output_ids = output_ids[:, mask]

        # Truncate at stop token if found
        if stop_token_ids is not None:
            stop_token_ids_tensor = torch.tensor(stop_token_ids, device=device)
            generated_tokens = output_ids[0, num_input_tokens:]
            stop_indices = torch.isin(generated_tokens, stop_token_ids_tensor).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_indices[0] + 1]

        stats.total_tokens = output_ids.shape[1] - num_input_tokens

        return output_ids, stats


def load_dflash_draft_model(
    draft_model_name_or_path: str = "/models/Qwen3-8B-DFlash-b16",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    dflash_path: str | Path | None = None,
    local_files_only: bool = True,
) -> nn.Module:
    """Load a DFlash draft model.

    Args:
        draft_model_name_or_path: HuggingFace model ID or local path.
        device: Device to load the model on.
        dtype: Data type for the model.
        dflash_path: Path to DFlash source code (for importing).
            Defaults to project_root/dflash.
        local_files_only: Only use local files (no HuggingFace download).

    Returns:
        The loaded DFlash draft model. The model may have a `mask_token_id` attribute
        if it was saved with the training config; use this value for DFlashConfig.mask_token_id_override.
    """
    # Add DFlash to path if needed
    if dflash_path is None:
        # Default: dflash folder is at project root (../../dflash from this file)
        dflash_path = Path(__file__).parent.parent.parent / "dflash"
    dflash_path = Path(dflash_path)

    if str(dflash_path) not in sys.path:
        sys.path.insert(0, str(dflash_path))

    from model.dflash import DFlashDraftModel

    logger.info(f"Loading DFlash draft model from {draft_model_name_or_path}")

    draft_model = DFlashDraftModel.from_pretrained(
        draft_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device).eval()

    # Log model info
    info_parts = [
        f"{draft_model.config.num_hidden_layers} layers",
        f"block_size={draft_model.block_size}",
    ]

    # Check for mask token ID in model config (important for correct setup)
    mask_token_id = getattr(draft_model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is not None:
        draft_model.mask_token_id = mask_token_id  # Ensure it's accessible
        info_parts.append(f"mask_token_id={mask_token_id}")
        logger.info(
            f"[DFlash] Found mask_token_id={mask_token_id} in checkpoint. "
            "Use this value for DFlashConfig.mask_token_id_override for best results."
        )

    logger.info(f"DFlash model loaded: {', '.join(info_parts)}")

    return draft_model


def create_dflash_accelerator(
    alpamayo_model: nn.Module,
    draft_model_name_or_path: str = "/models/Qwen3-8B-DFlash-b16",
    config: DFlashConfig | None = None,
    dflash_path: str | Path | None = None,
    local_files_only: bool = True,
) -> DFlashAlpamayoAccelerator:
    """Create a DFlash accelerator for an Alpamayo model.

    Convenience function that loads the draft model and creates the accelerator.
    Automatically configures trajectory token masking from the Alpamayo model config.

    Args:
        alpamayo_model: The Alpamayo model instance.
        draft_model_name_or_path: Path to the DFlash draft model.
        config: DFlash configuration. If None, will be auto-configured with
            trajectory token masking from the Alpamayo model.
        dflash_path: Path to DFlash source code.
        local_files_only: Only use local files (no HuggingFace download).

    Returns:
        Configured DFlashAlpamayoAccelerator.
    """
    device = next(alpamayo_model.parameters()).device
    dtype = next(alpamayo_model.parameters()).dtype

    # Load draft model
    draft_model = load_dflash_draft_model(
        draft_model_name_or_path,
        device=device,
        dtype=dtype,
        dflash_path=dflash_path,
        local_files_only=local_files_only,
    )

    # Auto-configure from Alpamayo and draft model if not provided
    if config is None:
        config = DFlashConfig()

    # Extract mask token ID from draft model (critical for correct embedding lookup)
    if config.mask_token_id_override is None:
        draft_mask_id = getattr(draft_model, "mask_token_id", None)
        if draft_mask_id is None:
            draft_mask_id = getattr(draft_model.config, "mask_token_id", None)
        if draft_mask_id is not None:
            config.mask_token_id_override = draft_mask_id
            logger.info(f"[DFlash] Auto-configured mask_token_id_override={draft_mask_id} from draft model")

    # Extract trajectory token configuration from Alpamayo model
    if config.traj_token_offset is None and hasattr(alpamayo_model, "config"):
        alpamayo_cfg = alpamayo_model.config
        if hasattr(alpamayo_cfg, "traj_token_start_idx"):
            config.traj_token_offset = alpamayo_cfg.traj_token_start_idx
            logger.info(f"[DFlash] Auto-configured traj_token_offset={config.traj_token_offset}")
        if hasattr(alpamayo_cfg, "traj_vocab_size"):
            config.traj_vocab_size = alpamayo_cfg.traj_vocab_size
            logger.info(f"[DFlash] Auto-configured traj_vocab_size={config.traj_vocab_size}")

    # Create accelerator
    accelerator = DFlashAlpamayoAccelerator(
        draft_model=draft_model,
        target_vlm=alpamayo_model.vlm,
        tokenizer=alpamayo_model.tokenizer,
        config=config,
    )

    # Load exact MASK embedding from training if available
    # This eliminates train-inference mismatch from re-computing vocab mean
    draft_path = Path(draft_model_name_or_path)
    mask_emb_file = draft_path / "mask_embedding.pt"

    if mask_emb_file.exists():
        logger.info(f"[DFlash] Loading exact mask embedding from {mask_emb_file}")
        mask_emb = torch.load(mask_emb_file, map_location=device)

        # Ensure dtype compatibility
        mask_emb = mask_emb.to(dtype=alpamayo_model.vlm.get_input_embeddings().weight.dtype)

        # Overwrite the mask embedding with the exact one from training
        with torch.no_grad():
            alpamayo_model.vlm.get_input_embeddings().weight[accelerator.mask_token_id] = mask_emb

        logger.info(f"[DFlash] Loaded exact mask embedding for token ID {accelerator.mask_token_id}")
    else:
        logger.warning(
            f"[DFlash] No mask_embedding.pt found in {draft_path}. "
            "Using vocabulary mean initialization (may cause train-inference mismatch)."
        )

    return accelerator
