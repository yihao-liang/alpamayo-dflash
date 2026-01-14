#!/usr/bin/env python3
"""Offline distillation: Train DFlash drafter on pre-computed hidden states.

Uses pre-generated distillation data (target_hidden, future_tokens, top_k_logits) for fast training.
No need to load Alpamayo during training - only DFlash is trained.

Supports:
- CE loss: Cross-entropy on actual tokens (default)
- KL loss: KL divergence using stored top-k logits
- CE+KL: Combined loss (recommended when top-k logits available)

Key optimization: Prefix-Weighted Cross-Entropy
- In speculative decoding, earlier tokens are more valuable (wrong first token = entire block rejected)
- Uses geometric decay weights: w(t) = exp(-t/gamma) to emphasize early positions
- Tracks first-token accuracy as critical diagnostic for inference acceptance

Usage (Single GPU):
    python train_dflash.py \
        --draft-model /models/Qwen3-8B-DFlash-b16 \
        --data-dir /data/dflash_distillation \
        --output-dir /exp/dflash_checkpoints \
        --loss-type ce+kl

Usage (Multi-GPU with torchrun):
    torchrun --nproc_per_node=8 train_dflash.py \
        --draft-model /models/Qwen3-8B-DFlash-b16 \
        --data-dir /data/dflash_distillation \
        --output-dir /exp/dflash_checkpoints \
        --loss-type ce+kl
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.dflash import DFlashDraftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Distributed Utilities ==============

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


# ============== Note on Training ==============
# DFlash uses discrete diffusion with MASK tokens, NOT continuous Gaussian noise.
# Training input: [context_token, MASK, MASK, ...]
# Training target: actual next tokens
# Loss: Cross-entropy on logits


# ============== Offline Dataset ==============

class ShardDataset(Dataset):
    """Dataset that loads a single shard of pre-computed distillation data."""

    def __init__(self, shard_path: str | Path):
        self.shard_path = Path(shard_path)
        data = torch.load(shard_path, map_location="cpu")
        self.target_hidden = data["target_hidden"]  # (num_blocks, hidden_dim) fp16
        self.future_tokens = data["future_tokens"]  # (num_blocks, block_size) int32
        # Labels with -100 for extended vocab tokens (vocab_size >= 151936)
        self.labels = data.get("labels", data["future_tokens"])  # fallback for old data

    def __len__(self):
        return self.target_hidden.shape[0]

    def __getitem__(self, idx):
        return {
            "target_hidden": self.target_hidden[idx],
            "future_tokens": self.future_tokens[idx],
            "labels": self.labels[idx],
        }


class OfflineDistillationDataset(Dataset):
    """Dataset that loads shards from a directory.

    Supports:
    - Per-GPU sharding (rank >= 0): Each GPU loads only its assigned shard(s)
    - Train/val split: Split each shard into train (90%) and val (10%)
    """

    def __init__(
        self,
        data_dir: str | Path,
        rank: int = -1,
        world_size: int = 1,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.rank = rank
        self.world_size = world_size
        self.split = split

        # Load metadata (try metadata.json first, then any metadata_rank*.json)
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            rank_metadata_files = sorted(self.data_dir.glob("metadata_rank*.json"))
            if rank_metadata_files:
                metadata_path = rank_metadata_files[0]
            else:
                raise FileNotFoundError(f"No metadata files found in {self.data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Per-GPU sharding: each GPU loads only its shard_rank{rank}_*.pt files
        if rank >= 0 and world_size > 1:
            # Explicitly match shard files by rank name
            my_shard_files = sorted(self.data_dir.glob(f"shard_rank{rank}_*.pt"))
            if not my_shard_files:
                raise FileNotFoundError(f"No shard files for rank {rank} in {data_dir}")
            logger.info(f"[Rank {rank}] Loading {len(my_shard_files)} shard(s): {[f.name for f in my_shard_files]}")
        else:
            # Single GPU or legacy mode: load all shards
            my_shard_files = sorted(self.data_dir.glob("shard_*.pt"))
            if not my_shard_files:
                raise FileNotFoundError(f"No shard files found in {data_dir}")
            if is_main_process():
                logger.info(f"Loading all {len(my_shard_files)} shards from {data_dir}...")

        # Load assigned shards
        all_hidden = []
        all_tokens = []
        all_labels = []

        all_topk_values = []
        all_topk_indices = []

        for shard_path in my_shard_files:
            data = torch.load(shard_path, map_location="cpu")
            all_hidden.append(data["target_hidden"])
            all_tokens.append(data["future_tokens"])
            # Labels with -100 for extended vocab tokens (vocab_size >= 151936)
            all_labels.append(data.get("labels", data["future_tokens"]))  # fallback for old data
            # Top-k logits for KL loss (optional, may not exist in old data)
            if "topk_values" in data:
                all_topk_values.append(data["topk_values"])
                all_topk_indices.append(data["topk_indices"])

        target_hidden = torch.cat(all_hidden, dim=0)
        future_tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Top-k logits (optional)
        if all_topk_values:
            topk_values = torch.cat(all_topk_values, dim=0)
            topk_indices = torch.cat(all_topk_indices, dim=0)
            self.has_topk_logits = True
        else:
            topk_values = None
            topk_indices = None
            self.has_topk_logits = False

        # Train/val split with fixed seed for reproducibility
        n_total = target_hidden.shape[0]
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        # Use fixed permutation so train/val splits are consistent
        generator = torch.Generator().manual_seed(seed + (rank if rank >= 0 else 0))
        perm = torch.randperm(n_total, generator=generator)

        if split == "train":
            indices = perm[:n_train]
        else:  # val
            indices = perm[n_train:]

        self.target_hidden = target_hidden[indices]
        self.future_tokens = future_tokens[indices]
        self.labels = labels[indices]

        # Top-k logits (optional)
        if self.has_topk_logits:
            self.topk_values = topk_values[indices]
            self.topk_indices = topk_indices[indices]
        else:
            self.topk_values = None
            self.topk_indices = None

        # Report stats
        log_rank = rank if rank >= 0 else 0
        logger.info(f"[Rank {log_rank}] {split}: {len(self)} blocks (from {n_total} total)")
        masked_count = (self.labels == -100).sum().item()
        total_count = self.labels.numel()
        if total_count > 0:
            logger.info(f"[Rank {log_rank}] {split}: masked tokens {masked_count}/{total_count} ({100*masked_count/total_count:.1f}%)")
        if self.has_topk_logits:
            logger.info(f"[Rank {log_rank}] {split}: top-k logits available (k={self.topk_values.shape[-1]})")

    def __len__(self):
        return self.target_hidden.shape[0]

    def __getitem__(self, idx):
        item = {
            "target_hidden": self.target_hidden[idx],
            "future_tokens": self.future_tokens[idx],
            "labels": self.labels[idx],
        }
        if self.has_topk_logits:
            item["topk_values"] = self.topk_values[idx]
            item["topk_indices"] = self.topk_indices[idx]
        return item


# ============== Trainer ==============

class OfflineDistillationTrainer:
    """Trainer for offline distillation using discrete MASK tokens and CE/KL loss.

    This matches the inference procedure:
    - Input: [context_token, MASK, MASK, ...]
    - DFlash predicts hidden states
    - lm_head converts to logits
    - Cross-entropy and/or KL loss

    Loss types:
    - ce: Cross-entropy on actual tokens (default)
    - kl: KL divergence using stored top-k logits
    - ce+kl: Combined loss (recommended when top-k logits available)
    """

    def __init__(
        self,
        draft_model: nn.Module,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        mask_token_id: int,
        block_size: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        local_rank: int = 0,
        world_size: int = 1,
        loss_type: str = "ce",
        ce_weight_start: float = 1.0,
        ce_weight_end: float = 0.8,
        kl_temperature: float = 1.0,
        prefix_weight_gamma: float = 1.5,
        target_layer_ids: list[int] | None = None,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

        # Draft model (DFlash) - trainable
        self.draft_model = draft_model.to(self.device)

        # Embedding layer from target model (frozen)
        self.embed_tokens = embed_tokens.to(self.device)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # LM head from target model (frozen, for computing logits)
        self.lm_head = lm_head.to(self.device)
        for param in self.lm_head.parameters():
            param.requires_grad = False

        self.mask_token_id = mask_token_id

        # Wrap with DDP if distributed
        if world_size > 1:
            self.draft_model = DDP(
                self.draft_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            self.model_without_ddp = self.draft_model.module
        else:
            self.model_without_ddp = self.draft_model

        self.block_size = block_size

        # Loss configuration
        self.loss_type = loss_type
        self.ce_weight_start = ce_weight_start
        self.ce_weight_end = ce_weight_end
        self.kl_temperature = kl_temperature
        self.prefix_weight_gamma = prefix_weight_gamma
        self.total_steps = total_steps

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.draft_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def get_lr(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def update_lr(self, total_steps: int):
        lr = self.get_lr(self.global_step, total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_ce_kl_weights(self) -> tuple[float, float]:
        """Get dynamic CE/KL weights based on training progress.

        Uses cosine schedule to transition from CE-dominated to balanced:
        - Start: ce_weight_start (e.g., 0.9), kl = 1 - ce
        - End: ce_weight_end (e.g., 0.5), kl = 1 - ce

        This allows the model to first learn hard targets (CE) then
        refine distribution matching (KL).
        """
        progress = min(1.0, self.global_step / max(1, self.total_steps))
        # Cosine schedule for smooth transition
        ce_weight = self.ce_weight_end + 0.5 * (self.ce_weight_start - self.ce_weight_end) * (1 + math.cos(math.pi * progress))
        kl_weight = 1.0 - ce_weight
        return ce_weight, kl_weight

    def compute_topk_kl_loss(
        self,
        draft_logits: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL loss using stored top-k target logits.

        Args:
            draft_logits: (B, seq, vocab_size) - draft model logits
            topk_values: (B, seq, k) - top-k target logit values
            topk_indices: (B, seq, k) - top-k target logit indices

        Returns:
            KL divergence loss
        """
        # Apply temperature
        if self.kl_temperature != 1.0:
            draft_logits = draft_logits / self.kl_temperature
            topk_values = topk_values / self.kl_temperature

        # Get draft logits at the same indices as target top-k
        draft_topk = torch.gather(draft_logits, dim=-1, index=topk_indices.long())

        # Compute softmax over top-k (approximation of full distribution)
        target_probs = F.softmax(topk_values, dim=-1)
        draft_log_probs = F.log_softmax(draft_topk, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(draft_log_probs, target_probs, reduction='batchmean', log_target=False)
        return kl_loss

    def train_step(self, batch: dict) -> dict:
        """Single training step using MASK tokens and CE/KL loss.

        This matches inference:
        1. Input: [first_token, MASK, MASK, ...] embeddings
        2. DFlash predicts hidden states for MASK positions
        3. lm_head converts to logits
        4. Cross-entropy and/or KL loss
        """
        self.draft_model.train()

        # Get model dtype
        model_dtype = next(self.model_without_ddp.parameters()).dtype

        target_hidden = batch["target_hidden"].to(self.device, dtype=model_dtype)  # (B, hidden_dim)
        future_tokens = batch["future_tokens"].to(self.device)  # (B, block_size)
        labels = batch["labels"].to(self.device, dtype=torch.long)  # (B, block_size) with -100 for extended vocab

        batch_size = target_hidden.shape[0]

        # Create masked input: [first_token, MASK, MASK, ...]
        # This matches inference where we know the first token and predict the rest
        masked_input = torch.full_like(future_tokens, self.mask_token_id)
        masked_input[:, 0] = future_tokens[:, 0]  # Keep first token (context)

        # Get embeddings of masked input
        with torch.no_grad():
            noise_embedding = self.embed_tokens(masked_input).to(dtype=model_dtype)  # (B, block_size, hidden)

        # Expand target_hidden for attention: (B, hidden_dim) -> (B, 1, hidden_dim)
        target_hidden_expanded = target_hidden.unsqueeze(1)

        # Position IDs: ctx (1) + block_size
        ctx_len = target_hidden_expanded.shape[1]
        full_seq_len = ctx_len + self.block_size
        position_ids = torch.arange(full_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Forward through DFlash
        draft_hidden = self.draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden_expanded,
            position_ids=position_ids,
            is_causal=False,
        )

        # Get logits for positions 1 to block_size (predicting MASK positions)
        # draft_hidden: (B, block_size, hidden) -> we want positions 1: (the MASK positions)
        logits = self.lm_head(draft_hidden[:, 1:, :])  # (B, block_size-1, vocab_size)

        # Labels for positions 1 to block_size (what the MASKs should predict)
        target_labels = labels[:, 1:]  # (B, block_size-1)

        # Compute losses
        metrics = {}

        # === Prefix-Weighted Cross-Entropy ===
        # In speculative decoding, earlier tokens are more valuable:
        # - Token 1 wrong = entire block discarded
        # - Token N only matters if all previous tokens are correct
        # Geometric decay weights emphasize early positions

        # 1. Compute raw CE per position (no reduction)
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none"
        ).view(target_labels.shape)  # (B, T)

        # 2. Construct geometric decay weights
        # gamma controls decay rate: higher = more emphasis on first token
        # gamma=1.5: weights ~[1.0, 0.51, 0.26, 0.13, 0.07, 0.03, 0.02, ...]
        T = target_labels.shape[1]
        step = torch.arange(T, device=self.device, dtype=logits.dtype)
        weights = torch.exp(-step / self.prefix_weight_gamma)
        weights = weights / weights.mean()  # Normalize to keep loss scale stable

        # 3. Apply weights and compute mean loss
        valid_mask = (target_labels != -100).float()
        valid_count = valid_mask.sum()
        if valid_count > 0:
            weighted_loss = (ce_per_token * weights[None, :] * valid_mask).sum() / valid_count
        else:
            weighted_loss = ce_per_token.mean()

        ce_loss = weighted_loss
        metrics["ce"] = ce_loss.item()

        # KL loss (if top-k logits available and KL enabled)
        kl_loss = torch.tensor(0.0, device=self.device)
        if "kl" in self.loss_type and "topk_values" in batch:
            topk_values = batch["topk_values"].to(self.device, dtype=model_dtype)
            topk_indices = batch["topk_indices"].to(self.device)
            kl_loss = self.compute_topk_kl_loss(logits, topk_values, topk_indices)
            metrics["kl"] = kl_loss.item()

        # Combined loss with dynamic CE/KL weights
        if self.loss_type == "ce":
            total_loss = ce_loss
            ce_weight, kl_weight = 1.0, 0.0
        elif self.loss_type == "kl":
            total_loss = kl_loss
            ce_weight, kl_weight = 0.0, 1.0
        elif self.loss_type == "ce+kl":
            ce_weight, kl_weight = self.get_ce_kl_weights()
            total_loss = ce_weight * ce_loss + kl_weight * kl_loss
        else:
            total_loss = ce_loss  # fallback
            ce_weight, kl_weight = 1.0, 0.0

        metrics["loss"] = total_loss.item()
        metrics["ce_weight"] = ce_weight
        metrics["kl_weight"] = kl_weight

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1

        # Compute accuracy for monitoring
        with torch.no_grad():
            valid_mask = target_labels != -100
            preds = logits.argmax(dim=-1)
            if valid_mask.sum() > 0:
                correct = (preds == target_labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

            # First-token accuracy - critical diagnostic for speculative decoding
            # If first token is wrong, entire block is rejected regardless of later tokens
            first_token_valid = valid_mask[:, 0]  # (B,)
            if first_token_valid.sum() > 0:
                first_token_correct = (preds[:, 0] == target_labels[:, 0]) & first_token_valid
                first_token_acc = first_token_correct.sum().float() / first_token_valid.sum().float()
            else:
                first_token_acc = torch.tensor(0.0)

            # Prefix accuracy - matches inference acceptance calculation
            # This measures consecutive correct tokens from position 0, like inference
            # For samples with -100 labels (trajectory tokens), treat as mismatch for prefix
            matches_for_prefix = (preds == target_labels) & (target_labels != -100)
            prefix_lengths = matches_for_prefix.cumprod(dim=1).sum(dim=1)  # (B,)
            prefix_rate = prefix_lengths.float().mean() / (self.block_size - 1)

        metrics["accuracy"] = accuracy.item()
        metrics["first_token_acc"] = first_token_acc.item()
        metrics["prefix_acc"] = prefix_rate.item()
        metrics["step"] = self.global_step
        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_steps: int,
        writer: SummaryWriter | None = None,
        log_interval: int = 10,
        max_batches: int | None = None,
    ) -> dict:
        """Train for one epoch."""
        total_loss = 0
        total_ce = 0
        total_kl = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=max_batches, disable=not is_main_process())
        for batch in pbar:
            # Limit batches to synchronized count across ranks
            if max_batches is not None and num_batches >= max_batches:
                break

            # Update learning rate
            lr = self.update_lr(total_steps)

            # Training step
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            total_ce += metrics.get("ce", 0)
            total_kl += metrics.get("kl", 0)
            num_batches += 1

            if is_main_process():
                postfix = {
                    "loss": f"{metrics['loss']:.4f}",
                    "t1": f"{metrics['first_token_acc']:.1%}",  # First-token accuracy - critical for acceptance
                    "pfx": f"{metrics['prefix_acc']:.1%}",  # Prefix accuracy - comparable to inference
                    "lr": f"{lr:.2e}"
                }
                if "kl" in metrics:
                    postfix["ce"] = f"{metrics['ce']:.4f}"
                    postfix["kl"] = f"{metrics['kl']:.4f}"
                    postfix["cw"] = f"{metrics['ce_weight']:.2f}"  # Dynamic CE weight
                pbar.set_postfix(postfix)

                # Log to tensorboard every log_interval steps
                if writer is not None and self.global_step % log_interval == 0:
                    writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    writer.add_scalar("train/ce_loss", metrics.get("ce", metrics["loss"]), self.global_step)
                    if "kl" in metrics:
                        writer.add_scalar("train/kl_loss", metrics["kl"], self.global_step)
                        writer.add_scalar("train/ce_weight", metrics["ce_weight"], self.global_step)
                        writer.add_scalar("train/kl_weight", metrics["kl_weight"], self.global_step)
                    writer.add_scalar("train/accuracy", metrics["accuracy"], self.global_step)
                    writer.add_scalar("train/first_token_accuracy", metrics["first_token_acc"], self.global_step)
                    writer.add_scalar("train/prefix_accuracy", metrics["prefix_acc"], self.global_step)
                    writer.add_scalar("train/lr", lr, self.global_step)

        # Average loss
        avg_loss = total_loss / max(num_batches, 1)
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        # Log epoch-level metrics
        if is_main_process() and writer is not None:
            writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        return {"epoch": epoch, "avg_loss": avg_loss, "total_steps": self.global_step}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, max_batches: int | None = None) -> dict:
        """Run validation with cross-entropy loss, accuracy, prefix accuracy, and first-token accuracy."""
        self.draft_model.eval()

        total_loss = 0
        total_correct = 0
        total_valid = 0
        total_first_correct = 0
        total_first_valid = 0
        total_prefix_len = 0
        total_samples = 0
        num_batches = 0

        model_dtype = next(self.model_without_ddp.parameters()).dtype

        for batch in dataloader:
            # Limit batches to synchronized count across ranks
            if max_batches is not None and num_batches >= max_batches:
                break
            target_hidden = batch["target_hidden"].to(self.device, dtype=model_dtype)
            future_tokens = batch["future_tokens"].to(self.device)
            labels = batch["labels"].to(self.device, dtype=torch.long)

            batch_size = target_hidden.shape[0]

            # Create masked input: [first_token, MASK, MASK, ...]
            masked_input = torch.full_like(future_tokens, self.mask_token_id)
            masked_input[:, 0] = future_tokens[:, 0]

            noise_embedding = self.embed_tokens(masked_input).to(dtype=model_dtype)

            target_hidden_expanded = target_hidden.unsqueeze(1)
            ctx_len = target_hidden_expanded.shape[1]
            full_seq_len = ctx_len + self.block_size
            position_ids = torch.arange(full_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

            draft_hidden = self.draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden_expanded,
                position_ids=position_ids,
                is_causal=False,
            )

            logits = self.lm_head(draft_hidden[:, 1:, :])
            target_labels = labels[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_labels.reshape(-1),
                ignore_index=-100,
            )

            # Per-token accuracy
            valid_mask = target_labels != -100
            preds = logits.argmax(dim=-1)
            if valid_mask.sum() > 0:
                correct = (preds == target_labels) & valid_mask
                total_correct += correct.sum().item()
                total_valid += valid_mask.sum().item()

            # First-token accuracy - critical diagnostic for speculative decoding
            first_token_valid = valid_mask[:, 0]
            if first_token_valid.sum() > 0:
                first_token_correct = (preds[:, 0] == target_labels[:, 0]) & first_token_valid
                total_first_correct += first_token_correct.sum().item()
                total_first_valid += first_token_valid.sum().item()

            # Prefix accuracy - matches inference acceptance calculation
            # Measures consecutive correct tokens from position 0
            matches_for_prefix = (preds == target_labels) & (target_labels != -100)
            prefix_lengths = matches_for_prefix.cumprod(dim=1).sum(dim=1)  # (B,)
            total_prefix_len += prefix_lengths.sum().item()
            total_samples += batch_size

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_valid, 1)
        first_token_accuracy = total_first_correct / max(total_first_valid, 1)
        prefix_accuracy = (total_prefix_len / max(total_samples, 1)) / (self.block_size - 1)

        # Average across all GPUs
        if self.world_size > 1:
            stats = torch.tensor(
                [avg_loss, total_correct, total_valid, total_prefix_len, total_samples,
                 total_first_correct, total_first_valid],
                device=self.device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_loss = stats[0].item() / self.world_size
            accuracy = stats[1].item() / max(stats[2].item(), 1)
            prefix_accuracy = (stats[3].item() / max(stats[4].item(), 1)) / (self.block_size - 1)
            first_token_accuracy = stats[5].item() / max(stats[6].item(), 1)

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_first_token_accuracy": first_token_accuracy,
            "val_prefix_accuracy": prefix_accuracy,
        }

    def save_checkpoint(self, path: str | Path, epoch: int, val_loss: float | None = None):
        if not is_main_process():
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Update model config with training parameters before saving
        # These are critical for correct inference
        self.model_without_ddp.config.block_size = self.block_size
        self.model_without_ddp.config.mask_token_id = self.mask_token_id

        self.model_without_ddp.save_pretrained(path)

        # Save exact MASK token embedding for inference
        # This eliminates train-inference mismatch from re-computing vocab mean
        mask_emb = self.embed_tokens.weight[self.mask_token_id].detach().cpu()
        torch.save(mask_emb, path / "mask_embedding.pt")

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if val_loss is not None:
            state["val_loss"] = val_loss

        torch.save(state, path / "training_state.pt")

        logger.info(f"Saved checkpoint to {path} (including mask_embedding.pt)")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Offline distillation: Train DFlash on pre-computed hidden states")
    parser.add_argument("--draft-model", type=str, default="/models/Qwen3-8B-DFlash-b16",
                        help="Path to pre-trained DFlash model for initialization")
    parser.add_argument("--target-model", type=str, default="/models/Alpamayo-R1-10B",
                        help="Path to target model (for embedding layer only)")
    parser.add_argument("--data-dir", type=str, default="/data/dflash_distillation",
                        help="Directory containing distillation data shards")
    parser.add_argument("--output-dir", type=str, default="/exp",
                        help="Base output directory (experiment folder with timestamp will be created)")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Base learning rate")
    parser.add_argument("--block-size", type=int, default=8,
                        help="Block size for DFlash (matches CoC avg length)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Warmup steps (default: 10%% of total steps)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory (default: output_dir/logs)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log to TensorBoard every N steps")
    # Loss configuration
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "kl", "ce+kl"],
                        help="Loss type: ce (cross-entropy), kl (KL div), ce+kl (combined)")
    parser.add_argument("--ce-weight-start", type=float, default=1.0,
                        help="Initial CE weight (for ce+kl with dynamic scheduling)")
    parser.add_argument("--ce-weight-end", type=float, default=0.8,
                        help="Final CE weight (for ce+kl with dynamic scheduling)")
    parser.add_argument("--kl-temperature", type=float, default=1.0,
                        help="Temperature for KL loss")
    # Prefix-weighted CE
    parser.add_argument("--prefix-weight-gamma", type=float, default=1.5,
                        help="Geometric decay coefficient for prefix-weighted CE (higher = more emphasis on early tokens)")
    # Vocabulary
    parser.add_argument("--full-vocab", action="store_true",
                        help="Resize draft model to full target vocab (155698) to support trajectory tokens")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Offline Distillation: DFlash Training")
        logger.info("=" * 60)
        logger.info(f"GPUs: {world_size}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Total batch size: {args.batch_size * world_size}")
        logger.info(f"Learning rate: {args.learning_rate}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    exp_name = f"dflash_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Output dir: {output_dir}")

    # Setup TensorBoard
    writer = None
    if is_main_process():
        log_dir = args.log_dir if args.log_dir else output_dir / "logs"
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logs: {log_dir}")

    if world_size > 1:
        dist.barrier()

    # Load DFlash model
    if is_main_process():
        logger.info(f"Loading DFlash from {args.draft_model}...")

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if is_main_process():
        logger.info(f"DFlash parameters: {sum(p.numel() for p in draft_model.parameters())/1e6:.1f}M")

    # Load embedding layer and lm_head from target model
    if is_main_process():
        logger.info(f"Loading target model from {args.target_model}...")

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    target_model = AlpamayoR1.from_pretrained(
        args.target_model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    embed_tokens = target_model.vlm.model.language_model.embed_tokens
    lm_head = target_model.vlm.lm_head
    tokenizer = target_model.tokenizer

    # Add MASK token if not present
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<|MASK|>'})
        # Resize embeddings to include the new token (disable automatic mean resizing
        # since we manually initialize the MASK embedding below)
        target_model.vlm.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # Initialize MASK embedding to mean of existing embeddings
        # This is critical: resize_token_embeddings randomly initializes new tokens,
        # and since we freeze embeddings, the MASK would otherwise be frozen random noise
        input_embeddings = target_model.vlm.get_input_embeddings()
        mask_token_id = tokenizer.mask_token_id
        with torch.no_grad():
            # Compute mean excluding the newly added token
            mean_embedding = input_embeddings.weight[:-1].mean(dim=0)
            input_embeddings.weight[mask_token_id] = mean_embedding
        if is_main_process():
            logger.info(f"Added MASK token (ID: {mask_token_id}), initialized embedding to vocab mean")

    mask_token_id = tokenizer.mask_token_id

    # Get target vocab size
    target_vocab_size = embed_tokens.weight.shape[0]
    draft_vocab_size = draft_model.config.vocab_size if hasattr(draft_model.config, 'vocab_size') else 151936

    if is_main_process():
        logger.info(f"Target vocab size: {target_vocab_size}")
        logger.info(f"Draft vocab size: {draft_vocab_size}")

    # Resize draft model vocab if needed (for full vocabulary training)
    if args.full_vocab and draft_vocab_size < target_vocab_size:
        if is_main_process():
            logger.info(f"Resizing draft model vocab: {draft_vocab_size} -> {target_vocab_size}")
        # DFlash model doesn't have its own embeddings (uses target's embed_tokens)
        # but we need to update the config for saving
        if hasattr(draft_model.config, 'vocab_size'):
            draft_model.config.vocab_size = target_vocab_size
        if is_main_process():
            logger.info("Draft model will use target's full vocabulary (including trajectory tokens)")

    # Free target model memory (keep only embed_tokens and lm_head)
    del target_model.vlm.model.language_model.layers
    del target_model.vlm.model.visual
    torch.cuda.empty_cache()

    if is_main_process():
        logger.info(f"MASK token ID: {mask_token_id}")
        if args.full_vocab:
            logger.info("Full vocab mode: Training on ALL tokens including trajectory tokens")

    # Load train/val datasets with per-GPU sharding (90/10 split)
    train_dataset = OfflineDistillationDataset(
        args.data_dir, rank=rank, world_size=world_size,
        split="train", val_ratio=0.1
    )
    val_dataset = OfflineDistillationDataset(
        args.data_dir, rank=rank, world_size=world_size,
        split="val", val_ratio=0.1
    )

    # Check if KL loss is requested but top-k logits not available
    if "kl" in args.loss_type and not train_dataset.has_topk_logits:
        if is_main_process():
            logger.warning("KL loss requested but top-k logits not found in data!")
            logger.warning("Falling back to CE loss only. Re-generate data with --top-k-logits to enable KL loss.")
        args.loss_type = "ce"

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Must match across ranks to avoid NCCL timeout at all_reduce
        persistent_workers=args.num_workers > 0,
    )

    # Sync after dataset loading
    if world_size > 1:
        dist.barrier()

    # Synchronize batch counts across ranks to avoid NCCL hangs
    # Each rank may have different dataset sizes due to per-GPU sharding
    local_train_batches = torch.tensor([len(train_loader)], device=f"cuda:{local_rank}", dtype=torch.int64)
    local_val_batches = torch.tensor([len(val_loader)], device=f"cuda:{local_rank}", dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(local_train_batches, op=dist.ReduceOp.MIN)
        dist.all_reduce(local_val_batches, op=dist.ReduceOp.MIN)
    max_train_batches = local_train_batches.item()
    max_val_batches = local_val_batches.item()

    if is_main_process():
        logger.info(f"Synchronized batch counts: train={max_train_batches}, val={max_val_batches}")

    # Calculate total steps (based on synchronized batch count)
    steps_per_epoch = max_train_batches
    total_steps = steps_per_epoch * args.num_epochs

    # Gather global stats
    local_train = torch.tensor([len(train_dataset)], device=f"cuda:{local_rank}", dtype=torch.int64)
    local_val = torch.tensor([len(val_dataset)], device=f"cuda:{local_rank}", dtype=torch.int64)
    if world_size > 1:
        all_train = [torch.zeros(1, device=f"cuda:{local_rank}", dtype=torch.int64) for _ in range(world_size)]
        all_val = [torch.zeros(1, device=f"cuda:{local_rank}", dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(all_train, local_train)
        dist.all_gather(all_val, local_val)
        total_train = sum(b.item() for b in all_train)
        total_val = sum(b.item() for b in all_val)
    else:
        total_train = local_train.item()
        total_val = local_val.item()

    # Auto-calculate warmup steps (10% of total) if not specified
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(10, total_steps // 10)

    # Scale learning rate with world size (linear scaling rule)
    # Effective batch size = batch_size * world_size, so LR scales proportionally
    scaled_lr = args.learning_rate * world_size

    if is_main_process():
        logger.info(f"Train blocks: {len(train_dataset):,} per GPU, {int(total_train):,} total")
        logger.info(f"Val blocks: {len(val_dataset):,} per GPU, {int(total_val):,} total")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps} ({'auto 10%' if args.warmup_steps is None else 'manual'})")
        logger.info(f"Learning rate: {args.learning_rate} x {world_size} GPUs = {scaled_lr}")

    # Create trainer (using MASK tokens + prefix-weighted cross-entropy loss)
    trainer = OfflineDistillationTrainer(
        draft_model=draft_model,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        learning_rate=scaled_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        local_rank=local_rank,
        world_size=world_size,
        loss_type=args.loss_type,
        ce_weight_start=args.ce_weight_start,
        ce_weight_end=args.ce_weight_end,
        kl_temperature=args.kl_temperature,
        prefix_weight_gamma=args.prefix_weight_gamma,
    )

    # Training loop
    if is_main_process():
        logger.info(f"Loss type: {args.loss_type} (prefix-weighted, gamma={args.prefix_weight_gamma})")
        if args.loss_type == "ce+kl":
            logger.info(f"CE/KL weights: {args.ce_weight_start:.1f} -> {args.ce_weight_end:.1f} (dynamic schedule)")
        logger.info(f"Starting offline distillation for {args.num_epochs} epochs...")

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(
            train_loader, epoch, total_steps,
            writer=writer, log_interval=args.log_interval,
            max_batches=max_train_batches,
        )

        # Validate
        val_metrics = trainer.validate(val_loader, max_batches=max_val_batches)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics.get("val_accuracy", 0)
        val_t1_acc = val_metrics.get("val_first_token_accuracy", 0)
        val_prefix_acc = val_metrics.get("val_prefix_accuracy", 0)

        if is_main_process():
            # Log first-token accuracy (critical), prefix accuracy (comparable to inference)
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['avg_loss']:.4f}, "
                f"val_loss={val_loss:.4f}, val_t1={val_t1_acc:.1%}, val_prefix={val_prefix_acc:.1%}"
            )

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/first_token_accuracy", val_t1_acc, epoch)
                writer.add_scalar("val/prefix_accuracy", val_prefix_acc, epoch)

        # Save checkpoint
        if epoch % args.save_every == 0:
            trainer.save_checkpoint(output_dir / f"checkpoint-epoch{epoch}", epoch, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(output_dir / "best", epoch, val_loss)
            if is_main_process():
                logger.info(f"  New best model! val_loss={val_loss:.4f}")

        if world_size > 1:
            dist.barrier()

    # Save final model
    trainer.save_checkpoint(output_dir / "final", args.num_epochs, val_loss)

    if is_main_process():
        with open(output_dir / "config.json", "w") as f:
            json.dump({
                "mode": "offline_distillation",
                "loss_type": args.loss_type,
                "prefix_weighted_ce": True,  # Geometric decay to emphasize early tokens
                "prefix_weight_gamma": args.prefix_weight_gamma,
                "ce_weight_start": args.ce_weight_start,
                "ce_weight_end": args.ce_weight_end,
                "draft_model": args.draft_model,
                "data_dir": args.data_dir,
                "block_size": args.block_size,
                "num_epochs": args.num_epochs,
                "base_learning_rate": args.learning_rate,
                "scaled_learning_rate": scaled_lr,
                "batch_size": args.batch_size,
                "world_size": world_size,
                "train_blocks": int(total_train),
                "val_blocks": int(total_val),
                "val_ratio": 0.1,
                "best_val_loss": best_val_loss,
                "mask_token_id": mask_token_id,
            }, f, indent=2)

        logger.info(f"Training complete! Best val_loss={best_val_loss:.4f}")
        logger.info(f"Model saved to {output_dir}")

        # Close TensorBoard writer
        if writer is not None:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
