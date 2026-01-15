#!/usr/bin/env python3
"""Train DFlash draft model from scratch for Alpamayo VLM.

This script creates a new DFlash model initialized with Alpamayo's vocabulary
and trains it using pre-computed distillation data.

Key differences from train_dflash.py:
- Creates a NEW DFlashDraftModel from scratch (not loading pre-trained)
- Uses Alpamayo's vocab_size (155,697) to support trajectory tokens
- Loads embed_tokens and lm_head from Alpamayo VLM
- Supports trainable embed_tokens/lm_head (optional)

Usage:
    # Train from scratch with frozen embeddings (recommended)
    torchrun --nproc_per_node=8 train_dflash_scratch.py \
        --target-model /models/Alpamayo-R1-10B \
        --data-dir /data/dflash_distillation \
        --output-dir /exp/dflash_scratch

    # Train with trainable embeddings
    torchrun --nproc_per_node=8 train_dflash_scratch.py \
        --target-model /models/Alpamayo-R1-10B \
        --data-dir /data/dflash_distillation \
        --output-dir /exp/dflash_scratch \
        --train-embeddings
"""

import argparse
import copy
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
from safetensors import safe_open

from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.dflash import DFlashDraftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Config Creation ==============

def make_draft_config_for_alpamayo(
    target_model_path: str,
    num_draft_layers: int = 5,
    block_size: int = 8,
    attn_implementation: str = "sdpa",
) -> Qwen3Config:
    """Create DFlash draft config from Alpamayo VLM config.

    Alpamayo is a VLM with extended vocabulary for trajectory tokens.
    This function reads the config.json and weights to extract necessary info.

    Args:
        target_model_path: Path to Alpamayo model
        num_draft_layers: Number of draft layers (default 5)
        block_size: Block size for speculative decoding (default 8)
        attn_implementation: Attention implementation (sdpa, flash_attention_2, eager)

    Returns:
        Qwen3Config configured for DFlash draft model
    """
    target_path = Path(target_model_path)

    # Load Alpamayo config.json directly (don't use AutoConfig which fails on custom models)
    config_path = target_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            alpamayo_config = json.load(f)
    else:
        alpamayo_config = {}

    # Get vocabulary size and hidden_size from embed_tokens weights (most reliable)
    index_path = target_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # Find embed_tokens weight to get actual vocab size
        for key, shard in index["weight_map"].items():
            if "embed_tokens" in key:
                shard_path = target_path / shard
                with safe_open(str(shard_path), framework="pt") as f:
                    emb_shape = f.get_tensor(key).shape
                vocab_size = emb_shape[0]
                hidden_size = emb_shape[1]
                logger.info(f"From weights: vocab_size={vocab_size}, hidden_size={hidden_size}")
                break
    else:
        # Fallback to config or defaults
        vocab_size = alpamayo_config.get('vocab_size', 155697)
        hidden_size = alpamayo_config.get('hidden_size', 4096)

    # Alpamayo-R1-10B is based on Qwen3-8B architecture
    # These are the known values for the language model part
    ALPAMAYO_LM_CONFIG = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": 36,  # Target model layers
        "hidden_act": "silu",
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
    }

    # Create draft config based on Qwen3 architecture
    draft_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=ALPAMAYO_LM_CONFIG["intermediate_size"],
        num_hidden_layers=num_draft_layers,
        num_attention_heads=ALPAMAYO_LM_CONFIG["num_attention_heads"],
        num_key_value_heads=ALPAMAYO_LM_CONFIG["num_key_value_heads"],
        head_dim=ALPAMAYO_LM_CONFIG["head_dim"],
        hidden_act=ALPAMAYO_LM_CONFIG["hidden_act"],
        max_position_embeddings=ALPAMAYO_LM_CONFIG["max_position_embeddings"],
        rms_norm_eps=ALPAMAYO_LM_CONFIG["rms_norm_eps"],
        rope_theta=ALPAMAYO_LM_CONFIG["rope_theta"],
        attention_bias=ALPAMAYO_LM_CONFIG["attention_bias"],
        attention_dropout=ALPAMAYO_LM_CONFIG["attention_dropout"],
        tie_word_embeddings=False,
    )

    # Add DFlash-specific config
    draft_config.num_target_layers = ALPAMAYO_LM_CONFIG["num_hidden_layers"]
    draft_config.block_size = block_size
    draft_config.max_window_layers = num_draft_layers
    draft_config._attn_implementation = attn_implementation
    draft_config.layer_types = ["full_attention"] * num_draft_layers

    return draft_config


def load_embed_lm_head_from_alpamayo(
    draft_model: nn.Module,
    target_model_path: str,
    device: torch.device = None,
) -> tuple[nn.Embedding, nn.Linear]:
    """Load embed_tokens and lm_head from Alpamayo VLM.

    Alpamayo stores these at:
    - vlm.model.language_model.embed_tokens.weight
    - vlm.lm_head.weight

    Args:
        draft_model: DFlash model (to get config)
        target_model_path: Path to Alpamayo model
        device: Device to load tensors to

    Returns:
        (embed_tokens, lm_head) modules initialized from Alpamayo
    """
    target_path = Path(target_model_path)
    index_path = target_path / "model.safetensors.index.json"

    # Weight keys for Alpamayo VLM
    EMB_KEY = "vlm.model.language_model.embed_tokens.weight"
    HEAD_KEY = "vlm.lm_head.weight"

    if index_path.exists():
        # Sharded model
        with open(index_path) as f:
            index = json.load(f)

        # Load embed_tokens
        emb_shard = index["weight_map"][EMB_KEY]
        with safe_open(str(target_path / emb_shard), framework="pt") as f:
            emb_weight = f.get_tensor(EMB_KEY)

        # Load lm_head
        head_shard = index["weight_map"][HEAD_KEY]
        with safe_open(str(target_path / head_shard), framework="pt") as f:
            head_weight = f.get_tensor(HEAD_KEY)
    else:
        # Single file model
        with safe_open(str(target_path / "model.safetensors"), framework="pt") as f:
            emb_weight = f.get_tensor(EMB_KEY)
            head_weight = f.get_tensor(HEAD_KEY)

    vocab_size, hidden_size = emb_weight.shape

    # Create modules
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    embed_tokens.weight.data.copy_(emb_weight)

    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    lm_head.weight.data.copy_(head_weight)

    if device is not None:
        embed_tokens = embed_tokens.to(device)
        lm_head = lm_head.to(device)

    logger.info(f"Loaded embed_tokens: {emb_weight.shape}, lm_head: {head_weight.shape}")

    return embed_tokens, lm_head


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


# ============== Dataset ==============

class OfflineDistillationDataset(Dataset):
    """Dataset that loads shards from a directory."""

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

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            rank_metadata_files = sorted(self.data_dir.glob("metadata_rank*.json"))
            if rank_metadata_files:
                metadata_path = rank_metadata_files[0]
            else:
                raise FileNotFoundError(f"No metadata files found in {self.data_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Per-GPU sharding
        if rank >= 0 and world_size > 1:
            my_shard_files = sorted(self.data_dir.glob(f"shard_rank{rank}_*.pt"))
            if not my_shard_files:
                raise FileNotFoundError(f"No shard files for rank {rank} in {data_dir}")
            logger.info(f"[Rank {rank}] Loading {len(my_shard_files)} shard(s)")
        else:
            my_shard_files = sorted(self.data_dir.glob("shard_*.pt"))
            if not my_shard_files:
                raise FileNotFoundError(f"No shard files found in {data_dir}")
            if is_main_process():
                logger.info(f"Loading all {len(my_shard_files)} shards from {data_dir}...")

        # Load shards
        all_hidden = []
        all_tokens = []
        all_labels = []
        all_topk_values = []
        all_topk_indices = []

        for shard_path in my_shard_files:
            data = torch.load(shard_path, map_location="cpu")
            all_hidden.append(data["target_hidden"])
            all_tokens.append(data["future_tokens"])
            all_labels.append(data.get("labels", data["future_tokens"]))
            if "topk_values" in data:
                all_topk_values.append(data["topk_values"])
                all_topk_indices.append(data["topk_indices"])

        target_hidden = torch.cat(all_hidden, dim=0)
        future_tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if all_topk_values:
            topk_values = torch.cat(all_topk_values, dim=0)
            topk_indices = torch.cat(all_topk_indices, dim=0)
            self.has_topk_logits = True
        else:
            topk_values = None
            topk_indices = None
            self.has_topk_logits = False

        # Train/val split
        n_total = target_hidden.shape[0]
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed + (rank if rank >= 0 else 0))
        perm = torch.randperm(n_total, generator=generator)

        if split == "train":
            indices = perm[:n_train]
        else:
            indices = perm[n_train:]

        self.target_hidden = target_hidden[indices]
        self.future_tokens = future_tokens[indices]
        self.labels = labels[indices]

        if self.has_topk_logits:
            self.topk_values = topk_values[indices]
            self.topk_indices = topk_indices[indices]
        else:
            self.topk_values = None
            self.topk_indices = None

        log_rank = rank if rank >= 0 else 0
        logger.info(f"[Rank {log_rank}] {split}: {len(self)} blocks")

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

class ScratchTrainer:
    """Trainer for training DFlash from scratch.

    Key features:
    - Supports trainable embed_tokens and lm_head
    - Uses prefix-weighted cross-entropy loss
    - Supports KL distillation with top-k logits
    """

    def __init__(
        self,
        draft_model: nn.Module,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        mask_token_id: int,
        block_size: int,
        train_embeddings: bool = False,
        learning_rate: float = 1e-4,
        embed_lr_scale: float = 0.1,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        local_rank: int = 0,
        world_size: int = 1,
        loss_type: str = "ce",
        prefix_weight_gamma: float = 1.5,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.train_embeddings = train_embeddings

        # Draft model
        self.draft_model = draft_model.to(self.device)

        # Embedding and LM head
        self.embed_tokens = embed_tokens.to(self.device)
        self.lm_head = lm_head.to(self.device)

        # Freeze or train embeddings
        if not train_embeddings:
            for param in self.embed_tokens.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            logger.info("embed_tokens and lm_head FROZEN")
        else:
            logger.info("embed_tokens and lm_head TRAINABLE")

        self.mask_token_id = mask_token_id
        self.block_size = block_size

        # Wrap with DDP if distributed
        if world_size > 1:
            self.draft_model = DDP(
                self.draft_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            self.model_without_ddp = self.draft_model.module

            if train_embeddings:
                self.embed_tokens = DDP(
                    self.embed_tokens,
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
                self.lm_head = DDP(
                    self.lm_head,
                    device_ids=[local_rank],
                    output_device=local_rank,
                )
        else:
            self.model_without_ddp = self.draft_model

        # Optimizer with separate param groups
        param_groups = [
            {"params": self.draft_model.parameters(), "lr": learning_rate},
        ]
        if train_embeddings:
            # Use smaller LR for embeddings to prevent divergence
            embed_lr = learning_rate * embed_lr_scale
            param_groups.extend([
                {"params": self.embed_tokens.parameters(), "lr": embed_lr},
                {"params": self.lm_head.parameters(), "lr": embed_lr},
            ])

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 0
        self.loss_type = loss_type
        self.prefix_weight_gamma = prefix_weight_gamma

    def get_lr(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def update_lr(self, total_steps: int):
        lr = self.get_lr(self.global_step, total_steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:
                param_group['lr'] = lr
            else:
                # Embeddings use scaled LR
                param_group['lr'] = lr * 0.1
        return lr

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.draft_model.train()
        if self.train_embeddings:
            self.embed_tokens.train()
            self.lm_head.train()

        model_dtype = next(self.model_without_ddp.parameters()).dtype

        target_hidden = batch["target_hidden"].to(self.device, dtype=model_dtype)
        future_tokens = batch["future_tokens"].to(self.device)
        labels = batch["labels"].to(self.device, dtype=torch.long)

        batch_size = target_hidden.shape[0]

        # Create masked input: [first_token, MASK, MASK, ...]
        masked_input = torch.full_like(future_tokens, self.mask_token_id)
        masked_input[:, 0] = future_tokens[:, 0]

        # Get embeddings
        if self.train_embeddings:
            noise_embedding = self.embed_tokens(masked_input).to(dtype=model_dtype)
        else:
            with torch.no_grad():
                noise_embedding = self.embed_tokens(masked_input).to(dtype=model_dtype)

        # Expand target_hidden for attention
        target_hidden_expanded = target_hidden.unsqueeze(1)

        # Position IDs
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

        # Get logits
        if self.train_embeddings:
            logits = self.lm_head(draft_hidden[:, 1:, :])
        else:
            with torch.no_grad():
                # Still need grad for draft_hidden, just not for lm_head weights
                pass
            logits = self.lm_head(draft_hidden[:, 1:, :])

        # Labels for positions 1 to block_size
        target_labels = labels[:, 1:]

        # Prefix-weighted cross-entropy
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none"
        ).view(target_labels.shape)

        T = target_labels.shape[1]
        step = torch.arange(T, device=self.device, dtype=logits.dtype)
        weights = torch.exp(-step / self.prefix_weight_gamma)
        weights = weights / weights.mean()

        valid_mask = (target_labels != -100).float()
        valid_count = valid_mask.sum()
        if valid_count > 0:
            ce_loss = (ce_per_token * weights[None, :] * valid_mask).sum() / valid_count
        else:
            ce_loss = ce_per_token.mean()

        total_loss = ce_loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), 1.0)
        if self.train_embeddings:
            torch.nn.utils.clip_grad_norm_(self.embed_tokens.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.lm_head.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1

        # Compute accuracy
        with torch.no_grad():
            valid_mask = target_labels != -100
            preds = logits.argmax(dim=-1)
            if valid_mask.sum() > 0:
                correct = (preds == target_labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

            # First-token accuracy
            first_token_valid = valid_mask[:, 0]
            if first_token_valid.sum() > 0:
                first_token_correct = (preds[:, 0] == target_labels[:, 0]) & first_token_valid
                first_token_acc = first_token_correct.sum().float() / first_token_valid.sum().float()
            else:
                first_token_acc = torch.tensor(0.0)

            # Prefix accuracy
            matches_for_prefix = (preds == target_labels) & (target_labels != -100)
            prefix_lengths = matches_for_prefix.cumprod(dim=1).sum(dim=1)
            prefix_rate = prefix_lengths.float().mean() / (self.block_size - 1)

        return {
            "loss": total_loss.item(),
            "ce": ce_loss.item(),
            "accuracy": accuracy.item(),
            "first_token_acc": first_token_acc.item(),
            "prefix_acc": prefix_rate.item(),
            "step": self.global_step,
        }

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
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=max_batches, disable=not is_main_process())
        for batch in pbar:
            if max_batches is not None and num_batches >= max_batches:
                break

            lr = self.update_lr(total_steps)
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1

            if is_main_process():
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "t1": f"{metrics['first_token_acc']:.1%}",
                    "pfx": f"{metrics['prefix_acc']:.1%}",
                    "lr": f"{lr:.2e}"
                })

                if writer is not None and self.global_step % log_interval == 0:
                    writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    writer.add_scalar("train/accuracy", metrics["accuracy"], self.global_step)
                    writer.add_scalar("train/first_token_accuracy", metrics["first_token_acc"], self.global_step)
                    writer.add_scalar("train/prefix_accuracy", metrics["prefix_acc"], self.global_step)
                    writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(num_batches, 1)
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return {"epoch": epoch, "avg_loss": avg_loss, "total_steps": self.global_step}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, max_batches: int | None = None) -> dict:
        """Run validation."""
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
            if max_batches is not None and num_batches >= max_batches:
                break

            target_hidden = batch["target_hidden"].to(self.device, dtype=model_dtype)
            future_tokens = batch["future_tokens"].to(self.device)
            labels = batch["labels"].to(self.device, dtype=torch.long)

            batch_size = target_hidden.shape[0]

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

            valid_mask = target_labels != -100
            preds = logits.argmax(dim=-1)
            if valid_mask.sum() > 0:
                correct = (preds == target_labels) & valid_mask
                total_correct += correct.sum().item()
                total_valid += valid_mask.sum().item()

            first_token_valid = valid_mask[:, 0]
            if first_token_valid.sum() > 0:
                first_token_correct = (preds[:, 0] == target_labels[:, 0]) & first_token_valid
                total_first_correct += first_token_correct.sum().item()
                total_first_valid += first_token_valid.sum().item()

            matches_for_prefix = (preds == target_labels) & (target_labels != -100)
            prefix_lengths = matches_for_prefix.cumprod(dim=1).sum(dim=1)
            total_prefix_len += prefix_lengths.sum().item()
            total_samples += batch_size

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_valid, 1)
        first_token_accuracy = total_first_correct / max(total_first_valid, 1)
        prefix_accuracy = (total_prefix_len / max(total_samples, 1)) / (self.block_size - 1)

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

        # Update config
        self.model_without_ddp.config.block_size = self.block_size
        self.model_without_ddp.config.mask_token_id = self.mask_token_id

        # Save draft model
        self.model_without_ddp.save_pretrained(path)

        # Save MASK embedding
        embed_module = self.embed_tokens.module if hasattr(self.embed_tokens, 'module') else self.embed_tokens
        mask_emb = embed_module.weight[self.mask_token_id].detach().cpu()
        torch.save(mask_emb, path / "mask_embedding.pt")

        # Save embed_tokens and lm_head if trainable
        if self.train_embeddings:
            lm_head_module = self.lm_head.module if hasattr(self.lm_head, 'module') else self.lm_head
            torch.save({
                "embed_tokens": embed_module.state_dict(),
                "lm_head": lm_head_module.state_dict(),
            }, path / "embeddings.pt")

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_embeddings": self.train_embeddings,
        }
        if val_loss is not None:
            state["val_loss"] = val_loss

        torch.save(state, path / "training_state.pt")
        logger.info(f"Saved checkpoint to {path}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Train DFlash from scratch for Alpamayo")
    parser.add_argument("--target-model", type=str, default="/models/Alpamayo-R1-10B",
                        help="Path to Alpamayo target model")
    parser.add_argument("--data-dir", type=str, default="/data/dflash_distillation",
                        help="Directory containing distillation data")
    parser.add_argument("--output-dir", type=str, default="/exp",
                        help="Base output directory")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Base learning rate")
    parser.add_argument("--block-size", type=int, default=8,
                        help="Block size (should match distillation data)")
    parser.add_argument("--num-draft-layers", type=int, default=5,
                        help="Number of draft layers")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Warmup steps (default: 10% of total)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    # Embedding training
    parser.add_argument("--train-embeddings", action="store_true",
                        help="Train embed_tokens and lm_head (default: frozen)")
    parser.add_argument("--embed-lr-scale", type=float, default=0.1,
                        help="Learning rate scale for embeddings (relative to base LR)")
    # Loss
    parser.add_argument("--prefix-weight-gamma", type=float, default=1.5,
                        help="Geometric decay for prefix-weighted CE")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Train DFlash from Scratch for Alpamayo")
        logger.info("=" * 60)
        logger.info(f"GPUs: {world_size}")
        logger.info(f"Train embeddings: {args.train_embeddings}")

    # Create output directory
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    mode = "trainable" if args.train_embeddings else "frozen"
    exp_name = f"dflash_scratch_{mode}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output: {output_dir}")

    # Setup TensorBoard
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    if world_size > 1:
        dist.barrier()

    # Create draft config from Alpamayo
    if is_main_process():
        logger.info(f"Creating draft config from {args.target_model}...")

    draft_config = make_draft_config_for_alpamayo(
        args.target_model,
        num_draft_layers=args.num_draft_layers,
        block_size=args.block_size,
    )

    if is_main_process():
        logger.info(f"Draft config: vocab_size={draft_config.vocab_size}, "
                    f"hidden_size={draft_config.hidden_size}, "
                    f"num_layers={draft_config.num_hidden_layers}")

    # Create DFlash model from scratch
    if is_main_process():
        logger.info("Creating DFlash model from scratch...")

    draft_model = DFlashDraftModel(draft_config)
    draft_model = draft_model.to(torch.bfloat16)

    if is_main_process():
        num_params = sum(p.numel() for p in draft_model.parameters())
        logger.info(f"DFlash parameters: {num_params/1e6:.1f}M")

    # Load embed_tokens and lm_head from Alpamayo
    if is_main_process():
        logger.info(f"Loading embed_tokens and lm_head from {args.target_model}...")

    embed_tokens, lm_head = load_embed_lm_head_from_alpamayo(
        draft_model, args.target_model
    )
    embed_tokens = embed_tokens.to(torch.bfloat16)
    lm_head = lm_head.to(torch.bfloat16)

    # Add MASK token
    # Alpamayo vocab is 155697, we add MASK at 155697
    vocab_size = embed_tokens.weight.shape[0]
    mask_token_id = vocab_size  # New token at end

    # Resize embeddings to add MASK token
    new_embed = nn.Embedding(vocab_size + 1, embed_tokens.weight.shape[1])
    new_embed.weight.data[:vocab_size] = embed_tokens.weight.data
    # Initialize MASK to mean of existing embeddings
    new_embed.weight.data[mask_token_id] = embed_tokens.weight.data.mean(dim=0)
    embed_tokens = new_embed.to(torch.bfloat16)

    new_lm_head = nn.Linear(lm_head.weight.shape[1], vocab_size + 1, bias=False)
    new_lm_head.weight.data[:vocab_size] = lm_head.weight.data
    new_lm_head.weight.data[mask_token_id] = lm_head.weight.data.mean(dim=0)
    lm_head = new_lm_head.to(torch.bfloat16)

    if is_main_process():
        logger.info(f"Added MASK token at ID {mask_token_id}")
        logger.info(f"Final vocab size: {vocab_size + 1}")

    # Load datasets
    train_dataset = OfflineDistillationDataset(
        args.data_dir, rank=rank, world_size=world_size,
        split="train", val_ratio=0.1
    )
    val_dataset = OfflineDistillationDataset(
        args.data_dir, rank=rank, world_size=world_size,
        split="val", val_ratio=0.1
    )

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
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    if world_size > 1:
        dist.barrier()

    # Synchronize batch counts
    local_train_batches = torch.tensor([len(train_loader)], device=f"cuda:{local_rank}", dtype=torch.int64)
    local_val_batches = torch.tensor([len(val_loader)], device=f"cuda:{local_rank}", dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(local_train_batches, op=dist.ReduceOp.MIN)
        dist.all_reduce(local_val_batches, op=dist.ReduceOp.MIN)
    max_train_batches = local_train_batches.item()
    max_val_batches = local_val_batches.item()

    steps_per_epoch = max_train_batches
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(10, total_steps // 10)

    # Scale LR with world size
    scaled_lr = args.learning_rate * world_size

    if is_main_process():
        logger.info(f"Train blocks: {len(train_dataset):,} per GPU")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Learning rate: {scaled_lr}")

    # Create trainer
    trainer = ScratchTrainer(
        draft_model=draft_model,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        train_embeddings=args.train_embeddings,
        learning_rate=scaled_lr,
        embed_lr_scale=args.embed_lr_scale,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        local_rank=local_rank,
        world_size=world_size,
        prefix_weight_gamma=args.prefix_weight_gamma,
    )

    # Training loop
    if is_main_process():
        logger.info(f"Starting training for {args.num_epochs} epochs...")

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = trainer.train_epoch(
            train_loader, epoch, total_steps,
            writer=writer, log_interval=args.log_interval,
            max_batches=max_train_batches,
        )

        val_metrics = trainer.validate(val_loader, max_batches=max_val_batches)
        val_loss = val_metrics["val_loss"]
        val_t1 = val_metrics["val_first_token_accuracy"]
        val_pfx = val_metrics["val_prefix_accuracy"]

        if is_main_process():
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['avg_loss']:.4f}, "
                f"val_loss={val_loss:.4f}, val_t1={val_t1:.1%}, val_pfx={val_pfx:.1%}"
            )

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/first_token_accuracy", val_t1, epoch)
                writer.add_scalar("val/prefix_accuracy", val_pfx, epoch)

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(output_dir / f"checkpoint-epoch{epoch}", epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(output_dir / "best", epoch, val_loss)
            if is_main_process():
                logger.info(f"  New best! val_loss={val_loss:.4f}")

        if world_size > 1:
            dist.barrier()

    # Save final
    trainer.save_checkpoint(output_dir / "final", args.num_epochs, val_loss)

    if is_main_process():
        with open(output_dir / "config.json", "w") as f:
            json.dump({
                "mode": "scratch",
                "train_embeddings": args.train_embeddings,
                "target_model": args.target_model,
                "num_draft_layers": args.num_draft_layers,
                "block_size": args.block_size,
                "vocab_size": vocab_size + 1,
                "mask_token_id": mask_token_id,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "scaled_learning_rate": scaled_lr,
                "batch_size": args.batch_size,
                "world_size": world_size,
                "best_val_loss": best_val_loss,
            }, f, indent=2)

        logger.info(f"Training complete! Best val_loss={best_val_loss:.4f}")
        logger.info(f"Model saved to {output_dir}")

        if writer is not None:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
