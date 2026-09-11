"""Data loading and preparation for the multitask bioactivity model.

The curated bioactivity tables (``data/bioactivity/<TARGET>.tsv``) contain raw
IC50 measurements in nM. Following the manuscript (Methods 2.2.1):

  * IC50 values are converted to pIC50: ``pIC50 = -log10(IC50 x 10^-9)``.
  * Structures are canonicalized with RDKit.
  * Duplicate measurements are consolidated by the median pIC50.
  * Two splits are supported: random (80/10/10) and Bemis-Murcko scaffold-based
    (primary evaluation).

KPGT embeddings are consumed from ``<features_dir>/<TARGET>.npz`` (key ``fps``,
shape ``(n, 2304)``), aligned row-wise with the deduplicated
``<TARGET>_dedup.csv`` produced by :func:`prepare_target`. Generate them with::

    python -m mtl.prepare --data-dir data/bioactivity --out-dir data/features/prepared
    cd kpgt/scripts
    python extract_features.py --config base --model_path <pretrained> \
        --data_path ../../data/features/prepared --device auto
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config


# --------------------------------------------------------------------------- #
# pIC50 conversion and deduplication
# --------------------------------------------------------------------------- #
def canonical_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string with RDKit ('' if invalid)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def to_pic50(ic50_nm) -> float:
    """Convert an IC50 value in nM to pIC50."""
    ic50 = float(ic50_nm)
    if ic50 <= 0.0 or not np.isfinite(ic50):
        raise ValueError(f"Invalid IC50 value: {ic50_nm}")
    return -np.log10(ic50 * 1e-9)


def load_bioactivity(tsv_path: str) -> pd.DataFrame:
    """Load a raw bioactivity TSV (Ligand SMILES / Target Name / IC50 (nM) / ...)."""
    df = pd.read_csv(tsv_path, sep="\t")
    smiles_col = "Ligand SMILES" if "Ligand SMILES" in df.columns else df.columns[0]
    ic50_col = "IC50 (nM)" if "IC50 (nM)" in df.columns else None
    df = df[[smiles_col] + ([ic50_col] if ic50_col else [])].copy()
    df.columns = ["smiles", "ic50_nm"] if ic50_col else ["smiles"]
    return df


def prepare_target(tsv_path: str, out_dir: str, target: str) -> pd.DataFrame:
    """Canonicalize, convert to pIC50, and consolidate duplicates by median.

    Saves ``<out_dir>/<TARGET>_dedup.csv`` with columns ``smiles`` and ``pIC50``.
    Returns the prepared DataFrame (one row per unique compound).
    """
    df = load_bioactivity(tsv_path)
    df["canonical"] = df["smiles"].map(canonical_smiles)
    df = df[df["canonical"] != ""].copy()

    if "ic50_nm" in df.columns:
        df = df[df["ic50_nm"].notna() & (df["ic50_nm"] > 0)].copy()
        df["pIC50"] = df["ic50_nm"].map(to_pic50)
    else:
        raise ValueError(f"No IC50 column found in {tsv_path}")

    # Consolidate duplicates: median pIC50 per canonical SMILES, keep first row.
    df = df.sort_values("canonical")
    agg = df.groupby("canonical")["pIC50"].median().reset_index()
    agg.columns = ["smiles", "pIC50"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{target}_dedup.csv")
    agg.to_csv(out_path, index=False)
    return agg


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _scaffold_of(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def split_indices(
    df: pd.DataFrame, strategy: str, seed: int, val_frac: float, test_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) for random or scaffold split."""
    rng = np.random.default_rng(seed)
    n = len(df)

    if strategy == "random":
        idx = rng.permutation(n)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        val_idx, test_idx = idx[:n_val], idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]
        return train_idx, val_idx, test_idx

    if strategy == "scaffold":
        scaffolds = df["smiles"].map(_scaffold_of).values
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(scaffolds):
            groups.setdefault(s, []).append(i)
        # Deterministic ordering: scaffold size desc, then scaffold id.
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        rng.shuffle(ordered)  # seed-controlled shuffling of scaffold groups
        train_idx, val_idx, test_idx = [], [], []
        n_target_val = int(round(n * val_frac))
        n_target_test = int(round(n * test_frac))
        # Assign largest scaffolds first to keep validation/test representative.
        for _, members in sorted(ordered, key=lambda kv: -len(kv[1])):
            if len(val_idx) < n_target_val:
                val_idx.extend(members)
            elif len(test_idx) < n_target_test:
                test_idx.extend(members)
            else:
                train_idx.extend(members)
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    raise ValueError(f"Unknown split strategy '{strategy}' (use 'random' or 'scaffold').")


# --------------------------------------------------------------------------- #
# Features and dataset
# --------------------------------------------------------------------------- #
def load_features(npz_path: str) -> np.ndarray:
    """Load a KPGT feature matrix from a NumPy archive (key ``fps`` if present)."""
    data = np.load(npz_path, allow_pickle=False)
    key = "fps" if "fps" in data else list(data.keys())[0]
    feats = np.asarray(data[key], dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix in {npz_path}, got {feats.shape}")
    return feats


class BioactivityDataset(Dataset):
    """Per-target dataset of (KPGT embedding, pIC50) pairs."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError(
                f"Feature/label mismatch: {len(features)} vs {len(labels)}"
            )
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.features[i], self.labels[i]


def build_task_data(
    config: Config, target: str
) -> Tuple[BioactivityDataset, BioactivityDataset, BioactivityDataset, pd.DataFrame]:
    """Prepare one target and return (train, val, test) datasets + metadata."""
    tsv_path = os.path.join(config.data_dir, f"{target}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Bioactivity table not found: {tsv_path}")

    df = prepare_target(tsv_path, config.prepared_dir, target)

    npz_path = os.path.join(config.features_dir, f"{target}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"KPGT embeddings not found: {npz_path}\n"
            "Generate them first (see mtl/README.md): run `python -m mtl.prepare`, then "
            "`kpgt/scripts/extract_features.py` on the prepared CSV."
        )
    feats = load_features(npz_path)
    if feats.shape[0] != len(df):
        raise ValueError(
            f"Feature rows ({feats.shape[0]}) do not match prepared compounds "
            f"({len(df)}) for {target}; re-extract features from the deduplicated CSV."
        )

    train_idx, val_idx, test_idx = split_indices(
        df, config.split, config.seed, config.val_frac, config.test_frac
    )
    labels = df["pIC50"].to_numpy()
    train_ds = BioactivityDataset(feats[train_idx], labels[train_idx])
    val_ds = BioactivityDataset(feats[val_idx], labels[val_idx])
    test_ds = BioactivityDataset(feats[test_idx], labels[test_idx])
    return train_ds, val_ds, test_ds, df


def build_all_task_data(
    config: Config,
) -> Dict[str, Tuple[BioactivityDataset, BioactivityDataset, BioactivityDataset, pd.DataFrame]]:
    """Prepare all configured targets."""
    return {t: build_task_data(config, t) for t in config.targets}
