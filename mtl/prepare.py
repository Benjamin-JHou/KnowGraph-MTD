"""Prepare bioactivity tables for feature extraction and training.

Canonicalizes SMILES, converts IC50 (nM) to pIC50, and consolidates duplicate
measurements by median pIC50. Writes ``<TARGET>_dedup.csv`` (smiles, pIC50) for
each target. The deduplicated CSVs are the input to
``kpgt/scripts/extract_features.py`` so that KPGT embeddings align row-wise
with the training labels.

Usage::

    python -m mtl.prepare --data-dir data/bioactivity --out-dir data/features/prepared
"""

import argparse
import os

from .config import Config
from .data import prepare_target


def main():
    parser = argparse.ArgumentParser(description="Prepare bioactivity tables (pIC50, dedup).")
    parser.add_argument("--data-dir", default=Config().data_dir)
    parser.add_argument("--out-dir", default=Config().prepared_dir)
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()

    targets = args.targets or Config().targets
    os.makedirs(args.out_dir, exist_ok=True)
    for t in targets:
        tsv = os.path.join(args.data_dir, f"{t}.tsv")
        df = prepare_target(tsv, args.out_dir, t)
        print(f"[prepare] {t}: {len(df)} unique compounds -> {args.out_dir}/{t}_dedup.csv")
    print("[prepare] Done. Next: run kpgt/scripts/extract_features.py on the prepared CSVs.")


if __name__ == "__main__":
    main()
