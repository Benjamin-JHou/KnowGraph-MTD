import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


# =========================================================
# Utils: load KPGT features (npz OR memmap)
# =========================================================
def load_kpgt_features(dir_path, prefix="kpgt_base"):
    npz_path = os.path.join(dir_path, f"{prefix}.npz")
    dat_path = os.path.join(dir_path, f"{prefix}.dat")
    shape_path = os.path.join(dir_path, f"{prefix}.shape.npy")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        feats = data["fps"] if "fps" in data else data[list(data.keys())[0]]
        print(f"[INFO] Loaded NPZ features: {npz_path}, shape={feats.shape}")
        return feats

    if os.path.exists(dat_path) and os.path.exists(shape_path):
        shape = np.load(shape_path)
        n, d = int(shape[0]), int(shape[1])
        feats = np.memmap(
            dat_path, dtype="float32", mode="r", shape=(n, d)
        )
        print(f"[INFO] Loaded MEMMAP features: {dat_path}, shape=({n}, {d})")
        return feats

    raise FileNotFoundError(f"No KPGT features found in {dir_path}")


# =========================================================
# Utils: load compound SMILES from unique CSV
# =========================================================
def load_compound_smiles(compound_dir):
    csv_files = [f for f in os.listdir(compound_dir) if f.endswith(".csv")]
    if len(csv_files) != 1:
        raise RuntimeError(
            f"Expected exactly ONE csv file in {compound_dir}, found {csv_files}"
        )

    csv_path = os.path.join(compound_dir, csv_files[0])
    df = pd.read_csv(csv_path, header=0)
    smiles = df.iloc[:, 0].astype(str).tolist()

    print(f"[INFO] Loaded SMILES from {csv_path}, total={len(smiles)}")
    return smiles


# =========================================================
# Model
# =========================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return nn.functional.normalize(x, dim=-1)


# =========================================================
# Contrastive loss
# =========================================================
def contrastive_loss(query, pos_proto, neg_protos, temperature=0.1):
    pos_sim = torch.matmul(query, pos_proto) / temperature
    neg_sim = torch.matmul(query, neg_protos.T) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    return nn.CrossEntropyLoss()(logits, labels)


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("MIL-Contrastive (enhanced output)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gene_dirs", nargs="+", required=True)
    parser.add_argument("--compound_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--aggregation", choices=["mean", "geometric", "min"], default="geometric")
    parser.add_argument("--final_topk", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="mil_results")
    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- Load features ----------------
    gene_features = {}
    for g in args.gene_dirs:
        gene_features[g] = load_kpgt_features(os.path.join(args.data_path, g))

    compound_dir = os.path.join(args.data_path, args.compound_dir)
    compound_feats = load_kpgt_features(compound_dir)
    compound_smiles = load_compound_smiles(compound_dir)

    feat_dim = compound_feats.shape[1]
    for g in gene_features:
        assert gene_features[g].shape[1] == feat_dim

    # ---------------- Model ----------------
    projector = ProjectionHead(feat_dim, args.proj_dim).to(device)
    optimizer = optim.Adam(projector.parameters(), lr=args.lr)
    gene_names = list(gene_features.keys())

    def compute_prototypes():
        protos = {}
        with torch.no_grad():
            for g in gene_names:
                x = torch.tensor(gene_features[g], dtype=torch.float32, device=device)
                z = projector(x)
                protos[g] = z.mean(dim=0)
        return protos

    # ---------------- Train ----------------
    print("[INFO] Training MIL-Contrastive")
    for epoch in range(1, args.epochs + 1):
        projector.train()
        total_loss = 0.0
        protos = compute_prototypes()

        for g in gene_names:
            pos_proto = protos[g]
            neg_protos = torch.stack([protos[x] for x in gene_names if x != g])

            feats = gene_features[g]
            idx = np.random.permutation(len(feats))

            for i in range(0, len(idx), args.batch_size):
                batch = torch.tensor(
                    feats[idx[i:i + args.batch_size]],
                    dtype=torch.float32,
                    device=device
                )
                optimizer.zero_grad()
                z = projector(batch)
                loss = contrastive_loss(z, pos_proto, neg_protos, args.temperature)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Loss = {total_loss:.4f}")

    # ---------------- Score compounds ----------------
    print("[INFO] Scoring compounds")
    projector.eval()
    protos = compute_prototypes()

    scores = {g: [] for g in gene_names}

    with torch.no_grad():
        for i in tqdm(range(len(compound_feats)), desc="Scoring"):
            x = torch.tensor(compound_feats[i], dtype=torch.float32, device=device).unsqueeze(0)
            z = projector(x).squeeze(0)
            for g in gene_names:
                scores[g].append(torch.dot(z, protos[g]).item())

    # ---------------- Aggregate ----------------
    agg_scores = []
    for i in range(len(compound_feats)):
        vals = np.array([scores[g][i] for g in gene_names])
        if args.aggregation == "mean":
            agg_scores.append(vals.mean())
        elif args.aggregation == "geometric":
            agg_scores.append(np.exp(np.mean(np.log(vals + 1e-8))))
        else:
            agg_scores.append(vals.min())

    agg_scores = np.array(agg_scores)
    topk_idx = np.argsort(-agg_scores)[:args.final_topk]

    # ---------------- Save results ----------------
    out_file = os.path.join(args.out_dir, f"projection_results.tsv")
    with open(out_file, "w") as f:
        header = ["compound_index", "smiles"]
        header += [f"score_{g}" for g in gene_names]
        header += ["agg_score", "aggregation"]
        f.write("\t".join(header) + "\n")

        for idx in topk_idx:
            row = [str(idx), compound_smiles[idx]]
            row += [f"{scores[g][idx]:.6f}" for g in gene_names]
            row += [f"{agg_scores[idx]:.6f}", args.aggregation]
            f.write("\t".join(row) + "\n")

    print(f"[OK] Final candidates saved to {out_file}")


if __name__ == "__main__":
    main()
