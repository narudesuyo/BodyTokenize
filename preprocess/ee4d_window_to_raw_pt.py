#!/usr/bin/env python3
"""
Slice EE4D kp3d clips into fixed-length overlapping windows.

Input .pt:
  {orig_key: {"kp3d": Tensor[T, J, 3], ...}, ...}

Output .pt:
  {new_key: {"kp3d": Tensor[clip_len, J, 3], "take": str, "start_frame": int, "end_frame": int}, ...}
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def make_windows(
    kp3d: np.ndarray,
    clip_len: int,
    overlap: int,
    include_tail: bool,
    pad_short: bool,
):
    if clip_len < 2:
        raise ValueError("clip_len must be >= 2")
    stride = clip_len - overlap
    if stride <= 0:
        raise ValueError("clip_len - overlap must be > 0")

    t = kp3d.shape[0]
    if t < 2:
        return []

    windows = []
    if t < clip_len:
        if not pad_short:
            return []
        pad = np.repeat(kp3d[-1:, :, :], clip_len - t, axis=0)
        clip = np.concatenate([kp3d, pad], axis=0)
        windows.append((clip, 0, t - 1))
        return windows

    starts = list(range(0, t - clip_len + 1, stride))
    if include_tail and starts[-1] != (t - clip_len):
        starts.append(t - clip_len)

    for s in starts:
        e = s + clip_len
        windows.append((kp3d[s:e], s, e - 1))
    return windows


def main():
    ap = argparse.ArgumentParser(
        description="Slice EE4D raw kp3d clips into fixed-length overlapping windows"
    )
    ap.add_argument("--in-pt", required=True, help="Input EE4D joints/tips .pt")
    ap.add_argument("--out-pt", required=True, help="Output raw windowed .pt")
    ap.add_argument("--clip-len", type=int, default=41)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--include-tail", dest="include_tail", action="store_true")
    ap.add_argument("--no-include-tail", dest="include_tail", action="store_false")
    ap.set_defaults(include_tail=True)
    ap.add_argument("--pad-short", dest="pad_short", action="store_true")
    ap.add_argument("--no-pad-short", dest="pad_short", action="store_false")
    ap.set_defaults(pad_short=True)
    ap.add_argument("--key-prefix", default="ee4d")
    ap.add_argument("--max-keys", type=int, default=-1)
    ap.add_argument("--max-clips", type=int, default=-1)
    args = ap.parse_args()

    in_pt = Path(args.in_pt)
    if not in_pt.exists():
        raise FileNotFoundError(f"input pt not found: {in_pt}")

    db_in = torch.load(str(in_pt), map_location="cpu", weights_only=False)
    if not isinstance(db_in, dict):
        raise TypeError(f"Expected dict in {in_pt}, got {type(db_in)}")

    keys = list(db_in.keys())
    if args.max_keys > 0:
        keys = keys[: args.max_keys]

    out = {}
    used_keys = 0
    total_windows = 0

    for k in tqdm(keys, desc="EE4D clips"):
        item = db_in.get(k)
        if not isinstance(item, dict) or "kp3d" not in item:
            continue

        kp3d = item["kp3d"]
        if torch.is_tensor(kp3d):
            kp3d = kp3d.detach().cpu().numpy()
        kp3d = np.asarray(kp3d)

        if kp3d.ndim != 3 or kp3d.shape[-1] != 3:
            continue

        wins = make_windows(
            kp3d=kp3d,
            clip_len=args.clip_len,
            overlap=args.overlap,
            include_tail=args.include_tail,
            pad_short=args.pad_short,
        )
        if not wins:
            continue

        for clip, f0, f1 in wins:
            new_key = f"{args.key_prefix}::{k}__{int(f0):06d}__{int(f1):06d}"
            out[new_key] = {
                "kp3d": torch.from_numpy(clip).float(),
                "take": str(k),
                "start_frame": int(f0),
                "end_frame": int(f1),
            }
            total_windows += 1
            if args.max_clips > 0 and total_windows >= args.max_clips:
                break

        used_keys += 1
        if args.max_clips > 0 and total_windows >= args.max_clips:
            break

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(out_pt))

    print(f"[DONE] input_keys={len(keys)}")
    print(f"[DONE] used_keys={used_keys}")
    print(f"[DONE] clips={len(out)}")
    print(f"[DONE] saved={out_pt}")


if __name__ == "__main__":
    main()

