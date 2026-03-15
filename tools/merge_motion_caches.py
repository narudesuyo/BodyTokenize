#!/usr/bin/env python3
"""Merge multiple precomputed motion cache .pt files into one.

Each input cache is expected to be:
  {key: {"body": Tensor[T,263], "hand": Tensor[T,480], ...}, ...}

The merged output keeps all entries and prefixes keys to avoid collisions.
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description="Merge motion cache .pt files")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input .pt paths")
    ap.add_argument(
        "--prefixes",
        nargs="+",
        default=None,
        help="Optional key prefixes (same length as --inputs).",
    )
    ap.add_argument("--output", required=True, help="Merged output .pt path")
    args = ap.parse_args()

    inputs = args.inputs
    if args.prefixes is None:
        prefixes = [f"src{i}" for i in range(len(inputs))]
    else:
        prefixes = args.prefixes
        if len(prefixes) != len(inputs):
            raise ValueError("--prefixes length must match --inputs length")

    merged = {}
    total_in = 0
    for src, pref in zip(inputs, prefixes):
        db = torch.load(src, map_location="cpu", weights_only=False)
        if not isinstance(db, dict):
            raise TypeError(f"Expected dict in {src}, got {type(db)}")

        print(f"[LOAD] {src}: {len(db)} samples (prefix={pref})")
        total_in += len(db)

        for k, v in db.items():
            new_k = f"{pref}::{k}"
            if new_k in merged:
                raise KeyError(f"Key collision after prefixing: {new_k}")
            merged[new_k] = v

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out))

    print(f"[DONE] input_total={total_in}")
    print(f"[DONE] merged={len(merged)}")
    print(f"[DONE] saved={out}")


if __name__ == "__main__":
    main()
