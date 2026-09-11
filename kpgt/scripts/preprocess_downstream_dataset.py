import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse
from tqdm import tqdm

from src.data.featurizer import smiles_to_graph_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=8)
    return parser.parse_args()


def preprocess_dataset(args):
    data_dir = f"{args.data_path}/{args.dataset}"
    df = pd.read_csv(f"{data_dir}/{args.dataset}.csv")

    smiless = df.smiles.tolist()
    task_names = df.columns.drop(['smiles']).tolist()

    # ======================================================
    # 1️⃣ Graph construction (唯一 valid_ids 来源)
    # ======================================================
    print("Constructing graphs...")
    graphs = pmap(
        smiles_to_graph_tune,
        smiless,
        max_length=args.path_length,
        n_virtual_nodes=2,
        n_jobs=args.n_jobs
    )

    valid_ids = []
    valid_graphs = []

    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    print(f"[INFO] Valid molecules after graph filtering: {len(valid_ids)} / {len(df)}")

    # ======================================================
    # 2️⃣ 对 csv / labels / smiles 做统一裁剪
    # ======================================================
    df = df.iloc[valid_ids].reset_index(drop=True)
    smiless = df.smiles.tolist()

    labels_np = df[task_names].values.astype(np.float32)
    labels = F.zerocopy_from_numpy(labels_np)

    # save graphs
    cache_file_path = f"{data_dir}/{args.dataset}_{args.path_length}.pkl"
    print("Saving graphs...")
    save_graphs(cache_file_path, valid_graphs, labels={'labels': labels})

    # ======================================================
    # 3️⃣ Fingerprints（不再单独判断 mol 是否 None）
    # ======================================================
    print("Extracting fingerprints...")
    FP_list = []

    for smi in tqdm(smiless):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, "Mol should not be None after graph filtering"
        fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)
        FP_list.append(fp)

    FP_mat = sp.csc_matrix(np.array(FP_list, dtype=np.int8))
    sp.save_npz(f"{data_dir}/rdkfp1-7_512.npz", FP_mat)

    # ======================================================
    # 4️⃣ Molecular descriptors（完全同一 smiles）
    # ======================================================
    print("Extracting molecular descriptors...")
    generator = RDKit2DNormalized()

    with Pool(args.n_jobs) as pool:
        features_map = list(
            tqdm(pool.imap(generator.process, smiless), total=len(smiless))
        )

    md_arr = np.array(features_map)[:, 1:]
    np.savez_compressed(
        f"{data_dir}/molecular_descriptors.npz",
        md=md_arr
    )

    # ======================================================
    # 5️⃣ 保存对齐后的 csv（可选但强烈推荐）
    # ======================================================
    df.to_csv(f"{data_dir}/{args.dataset}_filtered.csv", index=False)

    print("✅ Preprocessing finished successfully")
    print(f"✅ Final dataset size: {len(df)}")


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(args)
