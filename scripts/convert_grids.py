"""
Convert the 5 mpmaps pkl grids into per-key compressed npz slices.

Each slice is float32 and compressed, so a single slice is ~1 MB instead of
~50-400 MB for the full pkl. The webapp fetches only the slices it needs.

Run this once on the machine that hosts the pkl files (e.g. hephaistos), then
serve the resulting `slices/` directory over HTTPS. The webapp expects the
following layout under the configured base URL:

    slices/
      coordinates.npz                # Xmp, Ymp, Zmp, theta, phi
      bmsp_tilt{-30..30}.npz         # bx, by, bz
      bmsh_cone{1..90,12.5}.npz      # bx, by, bz
      nmsp_tilt{-30..30}.npz         # n
      nmsh_cone{1..90,12.5}.npz      # n
      manifest.json                  # available keys per family

Usage:
    python convert_grids.py [--src DIR] [--dst DIR]

By default, --src is the user-data-dir cache that mpmaps uses, and --dst is
./slices/ in the current working directory.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from platformdirs import user_data_dir


PKL_FILES = {
    "coordinates": "mp_coordinates.pkl",
    "bmsp":        "mp_b_msp.pkl",
    "bmsh":        "mp_b_msh.pkl",
    "nmsp":        "mp_np_msp.pkl",
    "nmsh":        "mp_np_msh.pkl",
}


def convert_coordinates(src_dir, dst_dir):
    d = pd.read_pickle(os.path.join(src_dir, PKL_FILES["coordinates"]))
    arrs = {k: v.astype(np.float32) for k, v in d.items()}
    out = os.path.join(dst_dir, "coordinates.npz")
    np.savez_compressed(out, **arrs)
    print(f"  → coordinates.npz ({os.path.getsize(out)/1e6:.2f} MB)")


def convert_b_dict(src_dir, dst_dir, family, prefix):
    """family is 'bmsp' or 'bmsh' (value is a (bx,by,bz) tuple)."""
    d = pd.read_pickle(os.path.join(src_dir, PKL_FILES[family]))
    keys = list(d.keys())
    print(f"  {family}: {len(keys)} slices")
    for key in keys:
        bx, by, bz = d[key]
        arrs = {
            "bx": bx.astype(np.float32),
            "by": by.astype(np.float32),
            "bz": bz.astype(np.float32),
        }
        out = os.path.join(dst_dir, f"{family}_{prefix}{key}.npz")
        np.savez_compressed(out, **arrs)
    return keys


def convert_n_dict(src_dir, dst_dir, family, prefix):
    """family is 'nmsp' or 'nmsh' (value is a single array)."""
    d = pd.read_pickle(os.path.join(src_dir, PKL_FILES[family]))
    keys = list(d.keys())
    print(f"  {family}: {len(keys)} slices")
    for key in keys:
        out = os.path.join(dst_dir, f"{family}_{prefix}{key}.npz")
        np.savez_compressed(out, n=d[key].astype(np.float32))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=os.path.join(user_data_dir(), "mpmaps"),
                        help="directory containing the .pkl grid files")
    parser.add_argument("--dst", default="./slices",
                        help="output directory for the .npz slices")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    print(f"src: {args.src}")
    print(f"dst: {args.dst}")

    convert_coordinates(args.src, args.dst)
    bmsp_keys = convert_b_dict(args.src, args.dst, "bmsp", "tilt")
    bmsh_keys = convert_b_dict(args.src, args.dst, "bmsh", "cone")
    nmsp_keys = convert_n_dict(args.src, args.dst, "nmsp", "tilt")
    nmsh_keys = convert_n_dict(args.src, args.dst, "nmsh", "cone")

    manifest = {
        "tilt_keys": sorted(bmsp_keys, key=lambda s: int(float(s))),
        "cone_keys": sorted(bmsh_keys, key=lambda s: float(s)),
    }
    with open(os.path.join(args.dst, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  → manifest.json")

    total = sum(os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst))
    print(f"\nTotal: {total/1e6:.1f} MB in {len(os.listdir(args.dst))} files")


if __name__ == "__main__":
    main()
