import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm

import sys
sys.path.append("..")

from src.utils import set_random_seed
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Extract molecular representations safely")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def extract_features(args):
    config = config_dict[args.config]

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_tune(config['path_length'])

    # datasets
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [
            d for d in sorted(os.listdir(args.data_path))
            if os.path.isdir(os.path.join(args.data_path, d))
        ]

    # model
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    for ds in datasets:
        print(f"\n[INFO] Processing dataset: {ds}")

        mol_dataset = MoleculeDataset(
            root_path=args.data_path,
            dataset=ds,
            dataset_type=None
        )

        loader = DataLoader(
            mol_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=args.pin_memory,
            drop_last=False
        )

        num_samples = len(mol_dataset)
        out_dir = os.path.join(args.data_path, ds)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"kpgt_{args.config}.dat")
        meta_path = os.path.join(out_dir, f"kpgt_{args.config}.shape.npy")

        fps_memmap = None
        offset = 0

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(
                tqdm(loader, desc=f"Extracting {ds}", total=len(loader))
            ):
                (_, g, ecfp, md, _) = batched_data

                g = g.to(device)
                ecfp = ecfp.to(device)
                md = md.to(device)

                fps = model.generate_fps(g, ecfp, md)
                fps = fps.detach().cpu().numpy()

                if fps_memmap is None:
                    feat_dim = fps.shape[1]
                    fps_memmap = np.memmap(
                        out_path,
                        dtype="float32",
                        mode="w+",
                        shape=(num_samples, feat_dim)
                    )

                batch_size = fps.shape[0]
                fps_memmap[offset:offset + batch_size] = fps
                offset += batch_size

        fps_memmap.flush()
        np.save(meta_path, np.array([num_samples, feat_dim]))

        print(f"[OK] Saved features to {out_path}")
        print(f"[OK] Shape metadata saved to {meta_path}")


if __name__ == "__main__":
    set_random_seed(22, 1)
    args = parse_args()
    extract_features(args)