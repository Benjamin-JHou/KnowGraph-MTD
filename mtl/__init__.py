"""Multitask bioactivity prediction module for KnowGraph-MTD.

Implements a multitask transformer over KPGT molecular embeddings:

  - shared transformer encoder,
  - task-specific residual-bottleneck adapter layers,
  - task-specific output heads,
  - dynamic weight averaging (DWA) for per-task loss reweighting,
  - gradient surgery (PCGrad-style) to resolve inter-task gradient conflicts.

The module is self-contained and consumes precomputed KPGT embeddings
(2,304-dimensional) produced by ``kpgt/scripts/extract_features.py``.
"""

__version__ = "0.1.0"
