"""Configuration for the multitask bioactivity model.

Default hyperparameters:
  - input projection: 2,304 -> 512 (LayerNorm + GELU + 15% dropout)
  - two transformer encoder blocks (d=512, FFN=2,048, 8 heads)
  - task adapters: residual bottleneck 512 -> 64 -> 512 (GELU)
  - task heads: 512 -> 256 -> 128 -> 1
  - AdamW: lr 1e-4 (shared) / 2e-4 (task heads), weight decay 1e-5
  - gradient clipping: max norm 1.0
  - ReduceLROnPlateau: factor 0.5, patience 5
  - early stopping: patience 15, up to 80 epochs, batch size 128
  - reproducibility seeds: {22, 42, 62, 82, 102}
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    data_dir: str = "data/bioactivity"      # directory with the 7 <TARGET>.tsv files
    features_dir: str = "data/features"     # directory with <TARGET>.npz KPGT embeddings
    prepared_dir: str = "data/features/prepared"  # deduplicated <TARGET>_dedup.csv files
    targets: List[str] = field(default_factory=lambda: [
        "ATRIP", "CSF1", "PDCD1", "SRC", "STAT3", "TLR7", "TNF",
    ])
    split: str = "scaffold"                 # "random" or "scaffold" (Bemis-Murcko)
    val_frac: float = 0.10
    test_frac: float = 0.10
    active_threshold: float = 5.0           # pIC50 >= 5.0 => active (for ROC/PR-AUC)

    # ----------------------------------------------------------- architecture
    input_dim: int = 2304                   # KPGT embedding dimension
    hidden_dim: int = 512
    ffn_dim: int = 2048
    n_heads: int = 8
    n_blocks: int = 2
    adapter_bottleneck: int = 64
    projection_dropout: float = 0.15
    transformer_dropout: float = 0.10
    head_dropout: float = 0.10

    # -------------------------------------------------------------- training
    batch_size: int = 128
    epochs: int = 80
    lr_shared: float = 1e-4                 # shared layers (projection + encoder + adapters)
    lr_head: float = 2e-4                   # task-specific heads
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 5
    early_stop_patience: int = 15
    num_workers: int = 0

    # ------------------------------------------------------- MTL components
    ablation: str = "full"  # full | no_dwa | no_gs | no_adapter | dwa_gs | dwa_adapter | gs_adapter | stl
    dwa_temperature: float = 2.0            # softmax temperature for DWA
    dwa_relative_change_clamp: float = 20.0
    gs_order: str = "desc"                  # gradient-surgery projection order: desc | asc | random

    # --------------------------------------------------------- reproducibility
    seed: int = 22
    seeds: Optional[List[int]] = None       # if set, overrides --seed (e.g. [22,42,62,82,102])
    device: str = "auto"                    # auto | cpu | cuda

    # ------------------------------------------------------------------ output
    out_dir: str = "results/mtl"


# Component flags derived from the ablation setting.
ABLATION_FLAGS = {
    "full":        dict(use_dwa=True,  use_gs=True,  use_adapters=True),
    "no_dwa":      dict(use_dwa=False, use_gs=True,  use_adapters=True),
    "no_gs":       dict(use_dwa=True,  use_gs=False, use_adapters=True),
    "no_adapter":  dict(use_dwa=True,  use_gs=True,  use_adapters=False),
    "dwa_gs":      dict(use_dwa=True,  use_gs=True,  use_adapters=False),
    "dwa_adapter": dict(use_dwa=True,  use_gs=False, use_adapters=True),
    "gs_adapter":  dict(use_dwa=False, use_gs=True,  use_adapters=True),
    "stl":         dict(use_dwa=False, use_gs=False, use_adapters=False),
}


def resolve_flags(ablation: str) -> dict:
    if ablation not in ABLATION_FLAGS:
        raise ValueError(
            f"Unknown ablation '{ablation}'. Choose from {sorted(ABLATION_FLAGS)}."
        )
    return dict(ABLATION_FLAGS[ablation])
