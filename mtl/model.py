"""Multitask transformer architecture.

Four sequential components:

1. ``InputProjection``  -- maps the 2,304-d KPGT embedding to a 512-d latent
   space with LayerNorm, GELU and 15% dropout.
2. ``TransformerBlock`` -- two stacked encoder blocks (model dim 512,
   feed-forward dim 2,048, 8-head self-attention).
3. ``Adapter``          -- seven task-specific residual bottlenecks
   (512 -> 64 -> 512, GELU).
4. ``TaskHead``         -- seven task-specific output heads
   (512 -> 256 -> 128 -> 1) producing scalar pIC50 predictions.

The total number of trainable parameters is ~9.8M (exact count is printed
by ``model.n_parameters``).
"""

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class InputProjection(nn.Module):
    """Project KPGT embeddings into the shared latent space."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(self.norm(self.linear(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with multi-head self-attention."""

    def __init__(self, hidden_dim: int, ffn_dim: int, n_heads: int, dropout: float = 0.10):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D); the model feeds a single-token sequence (L=1).
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class Adapter(nn.Module):
    """Task-specific residual bottleneck adapter (512 -> 64 -> 512, GELU)."""

    def __init__(self, hidden_dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class TaskHead(nn.Module):
    """Task-specific regression head (512 -> 256 -> 128 -> 1)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class MTLModel(nn.Module):
    """Multitask bioactivity model on KPGT embeddings.

    Parameters
    ----------
    config : Config
        Model configuration (architecture hyperparameters).
    task_names : List[str]
        Names of the prediction tasks (e.g. the seven target names).
    use_adapters : bool
        Whether to build task-specific adapter layers (ablation switch).
    """

    def __init__(self, config, task_names: List[str], use_adapters: bool = True):
        super().__init__()
        self.task_names = list(task_names)
        self.n_tasks = len(task_names)
        self.use_adapters = use_adapters

        self.projection = InputProjection(
            config.input_dim, config.hidden_dim, config.projection_dropout
        )
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                config.hidden_dim, config.ffn_dim, config.n_heads, config.transformer_dropout
            )
            for _ in range(config.n_blocks)
        ])
        if use_adapters:
            self.adapters = nn.ModuleList([
                Adapter(config.hidden_dim, config.adapter_bottleneck)
                for _ in range(self.n_tasks)
            ])
        else:
            self.adapters = None
        self.heads = nn.ModuleList([
            TaskHead(config.hidden_dim, config.head_dropout) for _ in range(self.n_tasks)
        ])

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Shared backbone: projection + stacked transformer blocks."""
        h = self.projection(x)
        h = h.unsqueeze(1)                      # (B, 1, D)
        h = self.blocks(h)                      # (B, 1, D)
        return h.squeeze(1)                     # (B, D)

    def forward_task(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        """Predict the pIC50 of task ``task_idx`` for the batch ``x``."""
        h = self.encode_shared(x)
        if self.adapters is not None:
            h = self.adapters[task_idx](h)
        return self.heads[task_idx](h)

    def forward(self, x: torch.Tensor, task_idx: Optional[int] = None) -> torch.Tensor:
        if task_idx is None:
            raise ValueError("MTLModel.forward requires task_idx; use forward_task instead.")
        return self.forward_task(x, task_idx)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_groups(self, lr_shared: float, lr_head: float, weight_decay: float):
        """Two parameter groups for differential learning rates."""
        shared = list(self.projection.parameters()) + list(self.blocks.parameters())
        if self.adapters is not None:
            shared += list(self.adapters.parameters())
        heads = list(self.heads.parameters())
        return [
            {"params": shared, "lr": lr_shared, "weight_decay": weight_decay},
            {"params": heads, "lr": lr_head, "weight_decay": weight_decay},
        ]
