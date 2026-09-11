# Multitask Bioactivity Model (mtl)

This module implements the **multitask transformer** described in Methods 2.2 of
the manuscript. It consumes precomputed **KPGT molecular embeddings**
(2,304 dimensions) and jointly predicts the pIC50 of compounds against the
seven autoimmune-relevant targets: ATRIP, CSF1, PDCD1, SRC, STAT3, TLR7, TNF.

## Architecture (Methods 2.2.2)

```
KPGT embedding (2,304)
        |
        v
Input projection (2,304 -> 512)  LayerNorm + GELU + dropout 0.15
        |
        v
2 x Transformer encoder block    d=512, FFN=2,048, 8 heads
        |
        +------------------+------------------+ ... +------------------+
        |                  |                  |      |                  |
    Adapter (0)        Adapter (1)        Adapter (2) ... Adapter (6)
    512->64->512        512->64->512        512->64->512 ...  512->64->512
        |                  |                  |      |                  |
    Head (0)            Head (1)            Head (2) ... Head (6)
    512->256->128->1    512->256->128->1    512->256->128->1 ... 512->256->128->1
        |                  |                  |      |                  |
     pIC50 (TNF)        pIC50 (STAT3)      pIC50 (TLR7) ... pIC50 (ATRIP)
```

* Shared backbone: input projection + two transformer encoder blocks.
* Task-specific residual-bottleneck adapters (512 -> 64 -> 512, GELU).
* Task-specific output heads (512 -> 256 -> 128 -> 1).
* Total trainable parameters: ~9.8M (exact count is printed at startup).

## Training components (Methods 2.2.3)

| Component | Implementation |
|---|---|
| Dynamic weight averaging (DWA) | Per-task loss weights updated from loss descent rates across consecutive epochs (`trainer.DynamicWeightAveraging`). |
| Gradient surgery (GS) | PCGrad-style projection: for each pair (i, j) with `cos(g_i, g_j) < 0`, remove task j's component from task i's gradient. Projection order follows descending task loss magnitude (configurable: `--gs-order desc/asc/random`). |
| Task adapters | Residual bottlenecks adding task-specific capacity on top of the shared encoder. |
| Optimizer | AdamW, differential learning rates: `1e-4` (shared layers) / `2e-4` (task heads), weight decay `1e-5`. |
| Gradient clipping | Max norm 1.0. |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5). |
| Early stopping | Patience 15 epochs, monitored on validation RMSE. |
| Training budget | Up to 80 epochs, batch size 128. |

## Data preparation

The curated bioactivity tables are in `../data/bioactivity/<TARGET>.tsv`
(columns: `Ligand SMILES`, `Target Name`, `IC50 (nM)`, `Curation/DataSource`,
`PDB ID(s) of Target Chain`). Following Methods 2.2.1:

1. Convert IC50 (nM) to pIC50: `pIC50 = -log10(IC50 x 10^-9)`.
2. Canonicalize SMILES with RDKit.
3. Consolidate duplicate measurements by the **median pIC50**.

```bash
# 1) Prepare deduplicated per-target CSVs (smiles, pIC50)
python -m mtl.prepare --data-dir data/bioactivity --out-dir data/features/prepared
```

## Generating KPGT embeddings

The model is trained on KPGT embeddings (2,304-d). Generate them with the KPGT
code shipped in `../kpgt` (needs a KPGT pretrained checkpoint, see
`../kpgt/README.md`):

```bash
cd kpgt/scripts
python extract_features.py --config base --model_path <pretrained_model_path> \
    --data_path ../../data/features/prepared --device auto
```

Place the resulting per-target archives as
`data/features/<TARGET>.npz` (key `fps`, shape `(n_compounds, 2304)`), aligned
row-wise with the corresponding `<TARGET>_dedup.csv`.

> The full-scale KPGT feature matrices for the seven training targets are large
> (tens of MB each) and are not shipped in this repository; they are available
> from the authors on request.

## Training

```bash
# Full model (DWA + GS + adapters), scaffold split, single seed
python -m mtl.train --data-dir data/bioactivity --features-dir data/features \
    --split scaffold --ablation full --seed 22

# Random split
python -m mtl.train --split random --ablation full --seed 22

# Five-seed reproducibility run (seeds {22, 42, 62, 82, 102})
python -m mtl.train --ablation full --seeds 22,42,62,82,102

# Ablation sweep (Table 2 of the manuscript)
python -m mtl.train --ablation no_dwa     --seeds 22,42,62,82,102
python -m mtl.train --ablation no_gs      --seeds 22,42,62,82,102
python -m mtl.train --ablation no_adapter --seeds 22,42,62,82,102
python -m mtl.train --ablation dwa_gs     --seeds 22,42,62,82,102
python -m mtl.train --ablation dwa_adapter --seeds 22,42,62,82,102
python -m mtl.train --ablation gs_adapter --seeds 22,42,62,82,102

# Single-task baseline (7 independent models, identical backbone)
python -m mtl.train --ablation stl --seeds 22,42,62,82,102
```

## Ablation modes

| `--ablation` | DWA | GS | Adapters | Notes |
|---|---|---|---|---|
| `full` | ✔ | ✔ | ✔ | Adopted configuration |
| `no_dwa` | ✘ | ✔ | ✔ | DWA removal |
| `no_gs` | ✔ | ✘ | ✔ | Gradient surgery removal |
| `no_adapter` | ✔ | ✔ | ✘ | Adapter removal |
| `dwa_gs` | ✔ | ✔ | ✘ | No adapters |
| `dwa_adapter` | ✔ | ✘ | ✔ | No gradient surgery |
| `gs_adapter` | ✘ | ✔ | ✔ | No DWA |
| `stl` | ✘ | ✘ | ✘ | Single-task per target |

## Evaluation metrics

Regression (Methods 2.2.3): RMSE, MAE, R², Pearson correlation coefficient.
Classification (compound active if pIC50 >= 5.0): ROC-AUC, PR-AUC.

Outputs are written to `--out-dir`:

* `<split>_<ablation>_seed<s>_metrics.json` — per-target and aggregate test metrics
* `<split>_<ablation>_seed<s>_predictions.csv` — pIC50 true/predicted per target
* `<split>_<ablation>_seed<s>_history.json` — training curves
* `<split>_<ablation>_summary.json` — mean / std / CV across seeds

## Reproducibility

Random seeds: `{22, 42, 62, 82, 102}` (the full pipeline was repeated five
times; all coefficients of variation remained below the predefined stability
threshold). Data partitioning: random split (80/10/10) and Bemis–Murcko
scaffold-based split (primary evaluation).

## Requirements

Python 3.12+, PyTorch 2.x, scikit-learn 1.7+, RDKit 2025.03+, NumPy, pandas.
