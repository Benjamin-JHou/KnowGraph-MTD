import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple KPGT memmap feature folders into one"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root path containing NC folders"
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="List of subdirectories to merge, e.g. NC1 NC2 NC3"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory name for merged features"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kpgt_base",
        help="Feature file prefix (default: kpgt_base)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- Collect shapes ----------------
    shapes = []
    total_n = 0
    feat_dim = None

    print("[INFO] Reading shapes...")
    for d in args.input_dirs:
        shape_path = os.path.join(args.data_path, d, f"{args.prefix}.shape.npy")
        if not os.path.exists(shape_path):
            raise FileNotFoundError(shape_path)

        n, d_dim = np.load(shape_path)
        n, d_dim = int(n), int(d_dim)

        if feat_dim is None:
            feat_dim = d_dim
        else:
            assert d_dim == feat_dim, \
                f"Feature dim mismatch: expected {feat_dim}, got {d_dim}"

        shapes.append((d, n))
        total_n += n
        print(f"  {d}: {n} × {d_dim}")

    print(f"[INFO] Total samples: {total_n}")
    print(f"[INFO] Feature dim: {feat_dim}")

    # ---------------- Prepare output ----------------
    out_dir = os.path.join(args.data_path, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_dat = os.path.join(out_dir, f"{args.prefix}.dat")
    out_shape = os.path.join(out_dir, f"{args.prefix}.shape.npy")

    print(f"[INFO] Creating merged memmap: {out_dat}")

    merged = np.memmap(
        out_dat,
        dtype="float32",
        mode="w+",
        shape=(total_n, feat_dim)
    )

    # ---------------- Merge ----------------
    offset = 0
    for d, n in shapes:
        dat_path = os.path.join(args.data_path, d, f"{args.prefix}.dat")
        shape_path = os.path.join(args.data_path, d, f"{args.prefix}.shape.npy")

        print(f"[INFO] Merging {d} ({n} samples)")
        src = np.memmap(
            dat_path,
            dtype="float32",
            mode="r",
            shape=(n, feat_dim)
        )

        merged[offset:offset + n] = src
        offset += n

        # explicitly flush each block
        merged.flush()
        del src

    assert offset == total_n
    np.save(out_shape, np.array([total_n, feat_dim]))

    print("[OK] Merge completed")
    print(f"[OK] Saved to: {out_dir}")
    print(f"[OK] Shape file: {out_shape}")


if __name__ == "__main__":
    main()
