import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# Utilities
# ===============================
def load_features(feat_dir, prefix="kpgt_base"):
    """
    Support:
    - .npz  (key: fps or arr)
    - .dat + .shape.npy (memmap)
    """
    npz = os.path.join(feat_dir, f"{prefix}.npz")
    dat = os.path.join(feat_dir, f"{prefix}.dat")
    shp = os.path.join(feat_dir, f"{prefix}.shape.npy")

    if os.path.exists(npz):
        data = np.load(npz)
        key = list(data.keys())[0]
        return data[key]

    if os.path.exists(dat) and os.path.exists(shp):
        shape = np.load(shp)
        return np.memmap(dat, dtype="float32", mode="r", shape=tuple(shape))

    raise FileNotFoundError(f"No feature file found in {feat_dir}")


def find_unique_csv(folder):
    csvs = glob(os.path.join(folder, "*.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"{folder} must contain exactly ONE csv file")
    return csvs[0]


def ic50_to_weight(ic50, alpha=1.0):
    """
    IC50-aware weighting (lower IC50 => higher weight)
    """
    ic50 = np.asarray(ic50).astype(float)
    w = np.exp(-alpha * np.log10(ic50 + 1.0))
    return w / w.sum()


# ===============================
# Main
# ===============================
def parse_args():
    parser = argparse.ArgumentParser("IC50-aware MIL Contrastive Screening")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gene_dirs", nargs="+", required=True)
    parser.add_argument("--compound_dir", type=str, required=True)
    parser.add_argument("--aggregation", type=str,
                        choices=["geometric", "mean", "min"],
                        default="geometric")
    parser.add_argument("--final_topk", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="mil_results")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="IC50 weight strength")
    return parser.parse_args()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load compounds ----------
    comp_feat_dir = os.path.join(args.data_path, args.compound_dir)
    comp_feats = load_features(comp_feat_dir)

    comp_csv = find_unique_csv(comp_feat_dir)
    comp_df = pd.read_csv(comp_csv)
    comp_smiles = comp_df.iloc[:, 0].values

    n_compounds = comp_feats.shape[0]

    # ---------- Process each target ----------
    gene_scores = {}
    gene_names = []

    for gene in args.gene_dirs:
        print(f"[INFO] Processing target: {gene}")
        gene_names.append(gene)

        gene_dir = os.path.join(args.data_path, gene)
        gene_feats = load_features(gene_dir)

        gene_csv = find_unique_csv(gene_dir)
        gene_df = pd.read_csv(gene_csv)

        ic50 = gene_df.iloc[:, 1].values  # second column = IC50 (nM)
        weights = ic50_to_weight(ic50, alpha=args.alpha)

        # cosine similarity: compounds × ligands
        sim = cosine_similarity(comp_feats, gene_feats)

        # IC50-aware MIL scoring
        score = (sim * weights[None, :]).sum(axis=1)
        gene_scores[gene] = score

    # ---------- Aggregate across targets ----------
    score_mat = np.stack([gene_scores[g] for g in gene_names], axis=1)

    if args.aggregation == "geometric":
        agg_score = np.exp(np.mean(np.log(score_mat + 1e-8), axis=1))
    elif args.aggregation == "mean":
        agg_score = score_mat.mean(axis=1)
    else:  # min
        agg_score = score_mat.min(axis=1)

    # ---------- Select top candidates ----------
    top_idx = np.argsort(-agg_score)[:args.final_topk]

    # ---------- Output ----------
    out_file = os.path.join(
        args.out_dir, f"ic50aware_results.tsv"
    )

    with open(out_file, "w") as f:
        header = ["compound_index", "smiles"]
        header += [f"score_{g}" for g in gene_names]
        header += ["agg_score", "aggregation"]
        f.write("\t".join(header) + "\n")

        for idx in top_idx:
            row = [
                str(idx),
                comp_smiles[idx]
            ]
            row += [f"{gene_scores[g][idx]:.6f}" for g in gene_names]
            row += [f"{agg_score[idx]:.6f}", args.aggregation]
            f.write("\t".join(row) + "\n")

    print(f"[OK] Final candidates saved to: {out_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
