#!/usr/bin/env python3
"""
Build a raw `.pt` database (`{key: {"kp3d": Tensor[T,J,3], ...}}`) from
Assembly101 `motion/v1/*/*.json` files.

This is compatible with `precompute.py --raw-pt ...` in this repository.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import smplx
from tqdm import tqdm


LEFT_TIP_VERTS = [5361, 4933, 5058, 5169, 5286]
RIGHT_TIP_VERTS = [8079, 7669, 7794, 7905, 8022]


def list_takes_from_split_csv(csv_path: str, motion_root: str = None) -> List[str]:
    videos = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row.get("video", "")
            if "/" in video:
                videos.add(video)
            elif video:
                videos.add(video)

    # Legacy behavior: return take ids.
    if motion_root is None:
        takes = {v.split("/", 1)[0] for v in videos}
        return sorted(takes)

    # Resolve split entries to actual motion directories.
    root = Path(motion_root)
    has_json_cache: Dict[str, bool] = {}

    def has_json(p: Path) -> bool:
        k = str(p)
        if k not in has_json_cache:
            has_json_cache[k] = p.is_dir() and any(p.glob("*.json"))
        return has_json_cache[k]

    resolved = []
    seen = set()
    unresolved = []

    for video in sorted(videos):
        take = video.split("/", 1)[0]
        candidates = [video, take]
        chosen = None

        for rel in candidates:
            if has_json(root / rel):
                chosen = rel
                break

        if chosen is None:
            # Fallback for nested structures where take has camera subdirs.
            take_dir = root / take
            if take_dir.is_dir():
                camera_dirs = sorted([p for p in take_dir.iterdir() if p.is_dir()])
                for cam in camera_dirs:
                    if has_json(cam):
                        chosen = str(Path(take) / cam.name)
                        break

        if chosen is None:
            unresolved.append(video)
            continue

        if chosen not in seen:
            seen.add(chosen)
            resolved.append(chosen)

    if unresolved:
        print(
            f"[WARN] Could not resolve {len(unresolved)} split entries under motion root. "
            f"Example: {unresolved[0]}"
        )

    return resolved


def list_all_takes(motion_root: str) -> List[str]:
    root = Path(motion_root)
    takes = set()
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # Flat layout: motion_root/<take>/*.json
        if any(p.glob("*.json")):
            takes.add(p.name)
            continue
        # Nested layout: motion_root/<take>/<camera>/*.json
        for c in p.iterdir():
            if c.is_dir() and any(c.glob("*.json")):
                takes.add(str(Path(p.name) / c.name))
    return sorted(takes)


def load_take_json_sequence(take_dir: str) -> Tuple[List[int], Dict[str, np.ndarray]]:
    frame_files = sorted(Path(take_dir).glob("*.json"), key=lambda p: int(p.stem))
    frame_ids = []
    betas = []
    transl = []
    global_orient = []
    body_pose = []
    left_hand_pose = []
    right_hand_pose = []

    for fp in frame_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        try:
            frame_ids.append(int(fp.stem))
            betas.append(np.asarray(obj["betas"], dtype=np.float32))
            transl.append(np.asarray(obj["transl"], dtype=np.float32))
            global_orient.append(np.asarray(obj["global_orient"], dtype=np.float32))
            body_pose.append(np.asarray(obj["body_pose"], dtype=np.float32))             # [21,3]
            left_hand_pose.append(np.asarray(obj["left_hand_pose"], dtype=np.float32))   # [15,3]
            right_hand_pose.append(np.asarray(obj["right_hand_pose"], dtype=np.float32)) # [15,3]
        except Exception:
            frame_ids.pop()
            continue

    if not frame_ids:
        return [], {}

    seq = {
        "betas": np.stack(betas, axis=0),  # [T,10] (often constant per frame)
        "transl": np.stack(transl, axis=0),  # [T,3]
        "global_orient": np.stack(global_orient, axis=0),  # [T,3] axis-angle
        "body_pose": np.stack(body_pose, axis=0),  # [T,21,3] axis-angle
        "left_hand_pose": np.stack(left_hand_pose, axis=0),  # [T,15,3] axis-angle
        "right_hand_pose": np.stack(right_hand_pose, axis=0),  # [T,15,3] axis-angle
    }
    return frame_ids, seq


def get_smplh_tip_vertex_ids():
    """Get SMPL-H fingertip vertex ids from smplx.vertex_ids when available."""
    try:
        from smplx.vertex_ids import vertex_ids
    except Exception:
        return None

    tip_names_l = ["lthumb", "lindex", "lmiddle", "lring", "lpinky"]
    tip_names_r = ["rthumb", "rindex", "rmiddle", "rring", "rpinky"]
    for model_key in ("smplh", "smplx"):
        vmap = vertex_ids.get(model_key)
        if vmap is None:
            continue
        try:
            tips_l = [vmap[n] for n in tip_names_l]
            tips_r = [vmap[n] for n in tip_names_r]
            return tips_l + tips_r
        except KeyError:
            continue
    return None


def joints52_to_kp3d154(joints_52: np.ndarray, tips_10: np.ndarray = None) -> np.ndarray:
    """Map SMPL-H 52 joints to EgoExo-style 154-joint kp3d layout."""
    T = joints_52.shape[0]
    kp3d = np.zeros((T, 154, 3), dtype=np.float32)
    kp3d[:, 0:22, :] = joints_52[:, 0:22, :]
    kp3d[:, 25:40, :] = joints_52[:, 22:37, :]
    kp3d[:, 40:55, :] = joints_52[:, 37:52, :]
    if tips_10 is not None:
        kp3d[:, -10:, :] = tips_10
    return kp3d


def build_body_model(
    model_dir: str,
    model_type: str,
    device: torch.device,
    gender: str,
    batch_size: int,
    flat_hand_mean: bool,
):
    # smplx.create(model_type="smplh") expects `model_path/smplh/SMPLH_*.pkl`.
    # Accept either:
    # - parent path (e.g. .../models/smplx), or
    # - direct smplh path (e.g. .../models/smplx/smplh).
    model_path = Path(model_dir)
    if model_type == "smplh":
        if model_path.name.lower() == "smplh":
            model_path = model_path.parent
        elif list(model_path.glob("SMPLH_*.pkl")) or list(model_path.glob("SMPLH_*.npz")):
            model_path = model_path.parent

    g = {"neutral": "NEUTRAL", "male": "MALE", "female": "FEMALE"}[gender.lower()]
    kwargs = {
        "model_path": str(model_path),
        "model_type": model_type,
        "gender": g,
        "batch_size": batch_size,
    }
    if model_type == "smplh":
        kwargs.update({"use_pca": False, "flat_hand_mean": flat_hand_mean})
    else:
        kwargs.update({"use_pca": False, "use_face_contour": True})

    layer = smplx.create(**kwargs).to(device)
    layer.eval()
    return layer


@torch.no_grad()
def smpl_seq_to_kp3d(
    seq: Dict[str, np.ndarray],
    model_dir: str,
    model_type: str,
    device: torch.device,
    gender: str,
    chunk_size: int,
    add_tips: bool,
    flat_hand_mean: bool,
    tip_vertex_ids: List[int],
) -> np.ndarray:
    T = seq["transl"].shape[0]
    chunks = []
    layer_cache = {}

    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        bsz = e - s
        if bsz not in layer_cache:
            layer_cache[bsz] = build_body_model(
                model_dir=model_dir,
                model_type=model_type,
                device=device,
                gender=gender,
                batch_size=bsz,
                flat_hand_mean=flat_hand_mean,
            )
        layer = layer_cache[bsz]

        betas = torch.from_numpy(seq["betas"][s:e]).to(device=device, dtype=torch.float32)
        transl = torch.from_numpy(seq["transl"][s:e]).to(device=device, dtype=torch.float32)
        global_orient = torch.from_numpy(seq["global_orient"][s:e]).to(device=device, dtype=torch.float32)
        body_pose = torch.from_numpy(seq["body_pose"][s:e].reshape(bsz, -1)).to(device=device, dtype=torch.float32)
        left_hand_pose = torch.from_numpy(seq["left_hand_pose"][s:e].reshape(bsz, -1)).to(device=device, dtype=torch.float32)
        right_hand_pose = torch.from_numpy(seq["right_hand_pose"][s:e].reshape(bsz, -1)).to(device=device, dtype=torch.float32)

        out = layer(
            betas=betas,
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose2rot=True,
        )

        if model_type == "smplh":
            joints_52 = out.joints[:, :52, :].detach().cpu().numpy().astype(np.float32)
            tips_10 = None
            if add_tips and tip_vertex_ids is not None:
                tips_10 = out.vertices[:, tip_vertex_ids, :].detach().cpu().numpy().astype(np.float32)
            kp = joints52_to_kp3d154(joints_52, tips_10=tips_10)
            chunks.append(kp)
        else:
            joints = out.joints
            if add_tips:
                tip_l = out.vertices[:, LEFT_TIP_VERTS, :]
                tip_r = out.vertices[:, RIGHT_TIP_VERTS, :]
                joints = torch.cat([joints, tip_l, tip_r], dim=1)
            chunks.append(joints.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(chunks, axis=0)


def make_windows(
    kp3d: np.ndarray,
    frame_ids: List[int],
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

    T = kp3d.shape[0]
    if T < 2:
        return []

    windows = []
    if T < clip_len:
        if not pad_short:
            return []
        pad = np.repeat(kp3d[-1:, :, :], clip_len - T, axis=0)
        clip = np.concatenate([kp3d, pad], axis=0)
        windows.append((clip, frame_ids[0], frame_ids[-1]))
        return windows

    starts = list(range(0, T - clip_len + 1, stride))
    if include_tail and starts[-1] != (T - clip_len):
        starts.append(T - clip_len)

    for s in starts:
        e = s + clip_len
        windows.append((kp3d[s:e], frame_ids[s], frame_ids[e - 1]))
    return windows


def main():
    ap = argparse.ArgumentParser(
        description="Convert Assembly101 motion/v1 JSONs into raw kp3d .pt clips"
    )
    ap.add_argument("--motion-root", required=True, help="e.g. /work/narus/data/Assembly101/motion/v1")
    ap.add_argument("--out-pt", required=True, help="Output .pt path")

    ap.add_argument("--text-root", default=None, help="e.g. /work/narus/data/Assembly101/text/v1")
    ap.add_argument("--split", default=None, choices=["train", "validation", "test"], help="Use takes from split CSV")

    ap.add_argument("--clip-len", type=int, default=41)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--include-tail", dest="include_tail", action="store_true")
    ap.add_argument("--no-include-tail", dest="include_tail", action="store_false")
    ap.set_defaults(include_tail=True)

    ap.add_argument("--pad-short", dest="pad_short", action="store_true")
    ap.add_argument("--no-pad-short", dest="pad_short", action="store_false")
    ap.set_defaults(pad_short=True)

    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--model-type", default="smplh", choices=["smplh", "smplx"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gender", default="male", choices=["neutral", "male", "female"])
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--flat-hand-mean", action="store_true")

    ap.add_argument("--add-tips", dest="add_tips", action="store_true")
    ap.add_argument("--no-add-tips", dest="add_tips", action="store_false")
    ap.set_defaults(add_tips=True)

    ap.add_argument("--downsample", type=int, default=1,
                     help="Temporal downsample factor (e.g. 3 for 30fps->10fps)")

    ap.add_argument("--key-prefix", default="a101")
    ap.add_argument("--max-takes", type=int, default=-1)
    ap.add_argument("--max-clips", type=int, default=-1)
    args = ap.parse_args()

    motion_root = Path(args.motion_root)
    if not motion_root.exists():
        raise FileNotFoundError(f"motion root not found: {motion_root}")

    if args.split is not None:
        if args.text_root is None:
            raise ValueError("--text-root is required when --split is set")
        split_csv = Path(args.text_root) / f"{args.split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"split csv not found: {split_csv}")
        takes = list_takes_from_split_csv(str(split_csv), motion_root=str(motion_root))
    else:
        takes = list_all_takes(str(motion_root))

    if args.max_takes > 0:
        takes = takes[: args.max_takes]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tip_vertex_ids = None
    if args.model_type == "smplh" and args.add_tips:
        tip_vertex_ids = get_smplh_tip_vertex_ids()
        if tip_vertex_ids is None:
            print("[WARN] SMPL-H tip vertices not found. Continuing without tips.")

    db = {}
    n_takes_ok = 0
    n_clips_total = 0

    for take in tqdm(takes, desc="Takes"):
        take_dir = motion_root / take
        if not take_dir.exists():
            continue

        frame_ids, seq = load_take_json_sequence(str(take_dir))
        if len(frame_ids) < 2:
            continue

        # downsample (e.g. 30fps -> 10fps with --downsample 3)
        if args.downsample > 1:
            frame_ids = frame_ids[::args.downsample]
            seq = {k: v[::args.downsample] for k, v in seq.items()}
            if len(frame_ids) < 2:
                continue

        kp3d = smpl_seq_to_kp3d(
            seq=seq,
            model_dir=args.model_dir,
            model_type=args.model_type,
            device=device,
            gender=args.gender,
            chunk_size=args.chunk_size,
            add_tips=args.add_tips,
            flat_hand_mean=args.flat_hand_mean,
            tip_vertex_ids=tip_vertex_ids,
        )

        wins = make_windows(
            kp3d=kp3d,
            frame_ids=frame_ids,
            clip_len=args.clip_len,
            overlap=args.overlap,
            include_tail=args.include_tail,
            pad_short=args.pad_short,
        )
        if not wins:
            continue

        for clip, f0, f1 in wins:
            key = f"{args.key_prefix}::{take}__{int(f0):06d}__{int(f1):06d}"
            db[key] = {
                "kp3d": torch.from_numpy(clip).float(),
                "take": take,
                "start_frame": int(f0),
                "end_frame": int(f1),
            }
            n_clips_total += 1
            if args.max_clips > 0 and n_clips_total >= args.max_clips:
                break

        n_takes_ok += 1
        if args.max_clips > 0 and n_clips_total >= args.max_clips:
            break

    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(db, str(out_pt))

    print(f"[DONE] takes_used={n_takes_ok}")
    print(f"[DONE] clips={len(db)}")
    print(f"[DONE] saved={out_pt}")


if __name__ == "__main__":
    main()
