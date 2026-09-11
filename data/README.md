# Data

This directory contains the data.

All file names are in English; column names follow the original curation. Precomputed feature matrices are stored as NumPy compressed archives (`.npz`).

## Layout

```
data/
├── bioactivity/            # Curated pIC50 training data (SMILES + experimental IC50)
├── targets/                # Target protein structures (PDB) and sequences (FASTA)
├── libraries/              # Natural product libraries for screening / external validation
└── representations/
    └── TCMNCs/             # Complete small example of precomputed features
```

---

## `bioactivity/`

Seven tab-separated files, one per target, containing the curated bioactivity measurements used to train and evaluate the multitask model.

| File | Compounds | Columns |
|---|---|---|
| `ATRIP.tsv` | 1,686 | `Ligand SMILES`, `Target Name`, `IC50 (nM)`, `Curation/DataSource`, `PDB ID(s) of Target Chain` |
| `CSF1.tsv` | 3,393 | same |
| `PDCD1.tsv` | 3,607 | same |
| `SRC.tsv` | 5,259 | same |
| `STAT3.tsv` | 1,197 | same |
| `TLR7.tsv` | 8,034 | same |
| `TNF.tsv` | 2,065 | same |

Notes:

- Compound counts are per target (a compound can appear under more than one target); the full multi-target dataset comprises 18,357 unique compounds.
- IC50 values are in nM; they are converted to pIC50 via `pIC50 = -log10(IC50 x 10^-9)`, deduplicated by canonical SMILES, and duplicate measurements are consolidated using the median pIC50.

## `targets/`

One directory per target protein (`ATRIP`, `CSF1`, `PDCD1`, `SRC`, `STAT3`, `TLR7`, `TNF`), each containing:

- `<TARGET>.pdb` — 3D structure of the target chain used for curation/reference.
- `<TARGET>.fasta` — amino-acid sequence of the target.

## `libraries/`

Natural product libraries used in screening and external validation.

| File | Compounds | Description |
|---|---|---|
| `HERB.csv` | 30,964 | HERB database natural product library used for multi-target screening; columns: `smiles`, `Ingredient_name` |
| `CMAUP.csv` | 2,979 | CMAUP natural product library (external validation); column: `smiles` |
| `TCMNCs.csv` | 740 | TCM natural compounds library (external validation); columns: `smiles`, `ScientificName`, `TCMMedicinesContaining` |
| `NCs_toy.csv` | 500 | Toy sample of the NCs natural compounds library (full set: 56,856 compounds) |

The screening workflow and Top results for the external validation libraries are described in the main `README.md` and in `../kpgt/scripts/mil_results/`.

## `representations/TCMNCs/`

A complete, small-scale example of the precomputed feature format consumed by the screening and scoring scripts (`ProjectionHead_MIL.py`, `IC50_aware_MIL_scoring.py`):

| File | Contents |
|---|---|
| `TCMNCs.csv` | Source SMILES (same as `../libraries/TCMNCs.csv`) |
| `kpgt_base.npz` | KPGT molecular embeddings (2,304 dimensions per compound) |
| `molecular_descriptors.npz` | Precomputed RDKit molecular descriptors |
| `rdkfp1-7_512.npz` | Precomputed RDKit fingerprints (radius 1–7, folded to 512 bits) |

## Full-scale data availability

The following resources are **not** included in this repository because of their size (individual files range from ~100 MB to several GB). They are available on request from the corresponding author:

- Full-scale KPGT feature matrices (`kpgt_base.npz` / `kpgt_base.dat` + `.shape.npy`) for the bioactivity training data, the HERB library, and the NCs/CMAUP/TCMNCs libraries.
- Precomputed molecular descriptors and fingerprints for the full libraries.
- Pickled processed datasets (`*_5.pkl`).
- Pretrained KPGT checkpoints and fine-tuned model weights (`*.pth`).

To reproduce the precomputed features yourself, extract them with `../kpgt/scripts/extract_features.py` using a KPGT pretrained model (see `../kpgt/README.md` for downloading the pretrained checkpoint).
