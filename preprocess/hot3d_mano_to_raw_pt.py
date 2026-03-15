#!/usr/bin/env python3
"""
HOT3D MANO → raw hand kp3d pt file.

Reads MANO hand pose trajectory + headset trajectory from HOT3D dataset,
runs MANO forward kinematics to get 3D hand joint positions (16 per hand),
and saves windowed clips as a pt file compatible with the hand-only pipeline.

Output format per key:
    {
        "kp3d_hands": Tensor[T, 32, 3],   # [16 LH joints, 16 RH joints] in world frame
        "head_pos": Tensor[T, 3],          # headset position
        "head_rot": Tensor[T, 4],          # headset rotation (wxyz quaternion)
    }

Requirements:
    pip install smplx
    MANO model files (.pkl) at --mano-model-dir
"""

import sys
import os
import json
import csv
import pickle
import argparse
import types
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# --- Mock chumpy to avoid installing it (needed for MANO pkl loading) ---
class _ChumpyArray(np.ndarray):
    """Mock chumpy array that extracts numpy data from chumpy pickle state."""
    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            return np.ndarray.__new__(cls, shape=(0,))
        return np.ndarray.__new__(cls, *args, **kwargs)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            super().__setstate__(state)
        elif isinstance(state, dict):
            if 'x' in state and hasattr(state['x'], 'shape'):
                arr = np.asarray(state['x'])
                self.resize(arr.shape, refcheck=False)
                np.copyto(self, arr)
            elif 'a' in state and 'idxs' in state:
                self.__dict__['_chumpy_a'] = state.get('a')
                self.__dict__['_chumpy_idxs'] = state.get('idxs')
                self.__dict__['_chumpy_shape'] = state.get('preferred_shape')
            else:
                self.__dict__.update(state)

class _CatchAll(types.ModuleType):
    def __getattr__(self, name):
        return _ChumpyArray

for _mod in ['chumpy', 'chumpy.ch', 'chumpy.utils', 'chumpy.reordering',
             'chumpy.linalg', 'chumpy.logic', 'chumpy.extras']:
    sys.modules[_mod] = _CatchAll(_mod)
# --- End chumpy mock ---

try:
    import smplx
except ImportError:
    print("ERROR: smplx not installed. Run: pip install smplx")
    sys.exit(1)

# Fingertip vertex indices in MANO mesh (778 vertices)
# From AffHandGen/common/mano_utils.py
MANO_FINGERTIP_INDICES = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky


def _resolve_chumpy(v):
    """Resolve chumpy Select objects to numpy arrays."""
    if isinstance(v, np.ndarray) and hasattr(v, '_chumpy_a'):
        a = _resolve_chumpy(v._chumpy_a)
        idxs = v._chumpy_idxs
        shape = v._chumpy_shape
        result = a.flat[idxs]
        if shape is not None:
            result = result.reshape(shape)
        return result
    return np.asarray(v) if isinstance(v, np.ndarray) else v


# HOT3D train/test split (Aria only, by participant ID)
TRAIN_PARTICIPANTS = {"P0001", "P0002", "P0003", "P0009", "P0010", "P0011", "P0012", "P0014", "P0015"}
TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}


def load_mano_trajectory(jsonl_path: str):
    """Load MANO hand pose trajectory from JSONL file.
    Returns list of dicts with keys: timestamp_ns, hand_poses.
    hand_poses: {hand_id: {pose: [15 PCA], wrist_xform: {t_xyz, q_wxyz}, betas: [10]}}
    """
    frames = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def load_headset_trajectory(csv_path: str):
    """Load headset trajectory from CSV.
    Returns dict: timestamp_ns → (pos_xyz, quat_wxyz).
    """
    traj = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp[ns]"])
            pos = np.array([
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ], dtype=np.float32)
            quat = np.array([
                float(row["q_wo_w"]),
                float(row["q_wo_x"]),
                float(row["q_wo_y"]),
                float(row["q_wo_z"]),
            ], dtype=np.float32)
            traj[ts] = (pos, quat)
    return traj


def load_hand_pose_mask(mask_dir: str):
    """Load mask_hand_pose_available.csv.
    Returns set of timestamps where hand pose is available.
    """
    mask_path = os.path.join(mask_dir, "mask_hand_pose_available.csv")
    if not os.path.exists(mask_path):
        return None  # no mask = all available

    available = set()
    with open(mask_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["mask"].strip().lower() == "true":
                available.add(int(row["timestamp[ns]"]))
    return available


def mano_fk(mano_layer, pose_pca, betas, wrist_xform, include_tips=False):
    """Run MANO forward kinematics.

    Args:
        mano_layer: smplx MANO model
        pose_pca: (15,) PCA coefficients
        betas: (10,) shape parameters
        wrist_xform: dict with t_xyz (3,) and q_wxyz (4,)
        include_tips: if True, append 5 fingertip positions from mesh vertices

    Returns:
        joints: (16, 3) or (21, 3) in world frame
    """
    device = next(mano_layer.parameters()).device

    hand_pose = torch.tensor(pose_pca, dtype=torch.float32, device=device).unsqueeze(0)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)

    # MANO FK in local (wrist) frame
    output = mano_layer(hand_pose=hand_pose, betas=betas_t)
    joints_local = output.joints[0].detach().cpu().numpy()  # (16, 3)

    if include_tips:
        # Extract fingertip positions from mesh vertices
        verts_local = output.vertices[0].detach().cpu().numpy()  # (778, 3)
        tips_local = verts_local[MANO_FINGERTIP_INDICES]  # (5, 3)
        joints_local = np.concatenate([joints_local, tips_local], axis=0)  # (21, 3)

    # Apply wrist transform to get world-frame joints
    t = np.array(wrist_xform["t_xyz"], dtype=np.float32)
    q = np.array(wrist_xform["q_wxyz"], dtype=np.float32)  # wxyz

    # Quaternion rotation
    joints_world = qrot_np_single(q, joints_local) + t
    return joints_world


def mano_fk_batch(mano_layer, poses_pca, betas_list, wrist_xforms, include_tips=False, batch_size=512):
    """Batched MANO forward kinematics for N frames at once.

    Args:
        mano_layer: smplx MANO model
        poses_pca: list of (15,) arrays, length N
        betas_list: list of (10,) arrays, length N
        wrist_xforms: list of dicts with t_xyz (3,) and q_wxyz (4,), length N
        include_tips: if True, append 5 fingertip positions
        batch_size: max batch size for GPU

    Returns:
        joints_world: (N, 16/21, 3) numpy array in world frame
    """
    device = next(mano_layer.parameters()).device
    N = len(poses_pca)

    all_joints = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs = end - start

        hand_pose = torch.tensor(np.array(poses_pca[start:end]), dtype=torch.float32, device=device)  # (bs, 15)
        betas_t = torch.tensor(np.array(betas_list[start:end]), dtype=torch.float32, device=device)    # (bs, 10)
        global_orient = torch.zeros(bs, 3, dtype=torch.float32, device=device)  # identity rotation

        with torch.no_grad():
            output = mano_layer(hand_pose=hand_pose, betas=betas_t, global_orient=global_orient)

        joints_local = output.joints.detach().cpu().numpy()  # (bs, 16, 3)

        if include_tips:
            verts_local = output.vertices.detach().cpu().numpy()  # (bs, 778, 3)
            tips_local = verts_local[:, MANO_FINGERTIP_INDICES]   # (bs, 5, 3)
            joints_local = np.concatenate([joints_local, tips_local], axis=1)  # (bs, 21, 3)

        # Apply per-frame wrist transforms
        for i in range(bs):
            idx = start + i
            t = np.array(wrist_xforms[idx]["t_xyz"], dtype=np.float32)
            q = np.array(wrist_xforms[idx]["q_wxyz"], dtype=np.float32)
            joints_local[i] = qrot_np_single(q, joints_local[i]) + t

        all_joints.append(joints_local)

    return np.concatenate(all_joints, axis=0)  # (N, 16/21, 3)


def qrot_np_single(q_wxyz, v):
    """Rotate vectors v by quaternion q (wxyz format).
    q: (4,), v: (N, 3) → (N, 3)
    """
    w, x, y, z = q_wxyz
    # quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return (R @ v.T).T


def window_sequence(data, clip_len, overlap):
    """Window a sequence into overlapping clips.
    data: dict with tensors of shape (T, ...) for each key.
    Returns list of windowed dicts.
    """
    T = next(iter(data.values())).shape[0]
    if T < clip_len:
        return [data]  # return as-is if shorter

    stride = clip_len - overlap
    windows = []
    for start in range(0, T - clip_len + 1, stride):
        window = {}
        for k, v in data.items():
            window[k] = v[start:start + clip_len]
        windows.append(window)
    return windows


def process_sequence(
    seq_dir: str,
    mano_left,
    mano_right,
    clip_len: int = 21,
    overlap: int = 10,
    include_tips: bool = False,
    downsample: int = 1,
):
    """Process a single HOT3D sequence.

    Returns dict of windowed clips: {key: {"kp3d_hands": (T,32/42,3), "head_pos": (T,3), "head_rot": (T,4)}}
    """
    seq_name = os.path.basename(seq_dir)
    mano_path = os.path.join(seq_dir, "mano_hand_pose_trajectory.jsonl")
    head_path = os.path.join(seq_dir, "headset_trajectory.csv")
    mask_dir = os.path.join(seq_dir, "masks")

    if not os.path.exists(mano_path) or not os.path.exists(head_path):
        return {}

    # Load data
    mano_frames = load_mano_trajectory(mano_path)
    head_traj = load_headset_trajectory(head_path)
    hand_mask = load_hand_pose_mask(mask_dir)

    # Build aligned arrays - collect all valid frames first, then batch FK
    head_timestamps = sorted(head_traj.keys())
    head_ts_arr = np.array(head_timestamps)

    # Collect valid frames
    lh_poses, lh_betas, lh_xforms = [], [], []
    rh_poses, rh_betas, rh_xforms = [], [], []
    all_head_pos = []
    all_head_rot = []
    all_lh_wrist_rot = []
    all_rh_wrist_rot = []

    for frame in mano_frames:
        ts = frame["timestamp_ns"]

        if hand_mask is not None and ts not in hand_mask:
            continue

        hp = frame.get("hand_poses", {})
        if "0" not in hp or "1" not in hp:
            continue

        idx = np.searchsorted(head_ts_arr, ts)
        idx = min(idx, len(head_ts_arr) - 1)
        if idx > 0 and abs(head_ts_arr[idx - 1] - ts) < abs(head_ts_arr[idx] - ts):
            idx = idx - 1
        matched_ts = head_ts_arr[idx]

        if abs(matched_ts - ts) > 100_000_000:
            continue

        head_pos, head_rot = head_traj[matched_ts]

        lh_poses.append(hp["0"]["pose"])
        lh_betas.append(hp["0"]["betas"])
        lh_xforms.append(hp["0"]["wrist_xform"])
        rh_poses.append(hp["1"]["pose"])
        rh_betas.append(hp["1"]["betas"])
        rh_xforms.append(hp["1"]["wrist_xform"])
        all_head_pos.append(head_pos)
        all_head_rot.append(head_rot)
        all_lh_wrist_rot.append(np.array(hp["0"]["wrist_xform"]["q_wxyz"], dtype=np.float32))
        all_rh_wrist_rot.append(np.array(hp["1"]["wrist_xform"]["q_wxyz"], dtype=np.float32))

    valid_count = len(lh_poses)
    if valid_count < clip_len:
        return {}

    # Batched MANO FK
    try:
        joints_lh_all = mano_fk_batch(mano_left, lh_poses, lh_betas, lh_xforms, include_tips=include_tips)
        joints_rh_all = mano_fk_batch(mano_right, rh_poses, rh_betas, rh_xforms, include_tips=include_tips)
    except Exception:
        return {}

    # Filter NaN frames
    valid_mask = ~(np.isnan(joints_lh_all).any(axis=(1, 2)) | np.isnan(joints_rh_all).any(axis=(1, 2)))
    joints_lh_all = joints_lh_all[valid_mask]
    joints_rh_all = joints_rh_all[valid_mask]
    all_head_pos = [all_head_pos[i] for i in range(valid_count) if valid_mask[i]]
    all_head_rot = [all_head_rot[i] for i in range(valid_count) if valid_mask[i]]
    all_lh_wrist_rot = [all_lh_wrist_rot[i] for i in range(valid_count) if valid_mask[i]]
    all_rh_wrist_rot = [all_rh_wrist_rot[i] for i in range(valid_count) if valid_mask[i]]
    valid_count = int(valid_mask.sum())

    if valid_count < clip_len:
        return {}

    # Stack into arrays
    all_joints = np.concatenate([joints_lh_all, joints_rh_all], axis=1)  # (T, 32/42, 3)
    all_head_pos = np.stack(all_head_pos, axis=0)  # (T, 3)
    all_head_rot = np.stack(all_head_rot, axis=0)  # (T, 4)
    all_lh_wrist_rot = np.stack(all_lh_wrist_rot, axis=0)  # (T, 4) wxyz
    all_rh_wrist_rot = np.stack(all_rh_wrist_rot, axis=0)  # (T, 4) wxyz

    # Temporal downsample (e.g. 3 for 30fps -> 10fps)
    if downsample > 1:
        all_joints = all_joints[::downsample]
        all_head_pos = all_head_pos[::downsample]
        all_head_rot = all_head_rot[::downsample]
        all_lh_wrist_rot = all_lh_wrist_rot[::downsample]
        all_rh_wrist_rot = all_rh_wrist_rot[::downsample]

    # Window into clips
    full_data = {
        "kp3d_hands": all_joints,
        "head_pos": all_head_pos,
        "head_rot": all_head_rot,
        "wrist_rot_lh": all_lh_wrist_rot,
        "wrist_rot_rh": all_rh_wrist_rot,
    }

    windows = window_sequence(full_data, clip_len, overlap)

    result = {}
    for i, win in enumerate(windows):
        key = f"{seq_name}::clip_{i:04d}"
        result[key] = {
            "kp3d_hands": torch.from_numpy(win["kp3d_hands"]).float(),
            "head_pos": torch.from_numpy(win["head_pos"]).float(),
            "head_rot": torch.from_numpy(win["head_rot"]).float(),
            "wrist_rot_lh": torch.from_numpy(win["wrist_rot_lh"]).float(),
            "wrist_rot_rh": torch.from_numpy(win["wrist_rot_rh"]).float(),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="HOT3D MANO → raw kp3d pt file")
    parser.add_argument("--data-dir", type=str, default="/work/narus/data/HOT3D/",
                        help="HOT3D dataset root directory")
    parser.add_argument("--mano-model-dir", type=str, required=True,
                        help="Directory containing MANO model files (MANO_LEFT.pkl, MANO_RIGHT.pkl)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for pt files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"],
                        help="Dataset split to process")
    parser.add_argument("--clip-len", type=int, default=21,
                        help="Clip length in frames")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Overlap between consecutive clips")
    parser.add_argument("--num-pca-comps", type=int, default=15,
                        help="Number of MANO PCA components")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for MANO FK (cpu or cuda)")
    parser.add_argument("--include-tips", action="store_true",
                        help="Extract fingertip positions from MANO mesh vertices (21 joints/hand instead of 16)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Temporal downsample factor (e.g. 3 for 30fps->10fps)")
    args = parser.parse_args()

    # Load MANO models
    # Pre-load and resolve chumpy objects in pkl, then save clean pkl for smplx
    import tempfile
    print("Loading MANO models...")
    mano_dir = Path(args.mano_model_dir)

    def load_resolved_mano(pkl_path, is_rhand, num_pca_comps, device):
        """Load MANO pkl, resolve chumpy objects, and create smplx model."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        for k in list(data.keys()):
            data[k] = _resolve_chumpy(data[k])
        # Save resolved pkl to temp file for smplx (name must contain MANO)
        suffix = '_RIGHT.pkl' if is_rhand else '_LEFT.pkl'
        with tempfile.NamedTemporaryFile(prefix='MANO', suffix=suffix, delete=False) as tmp:
            pickle.dump(data, tmp)
            tmp_path = tmp.name
        try:
            model = smplx.create(
                tmp_path, model_type="mano", is_rhand=is_rhand,
                use_pca=True, num_pca_comps=num_pca_comps, flat_hand_mean=False,
            ).to(device)
        finally:
            os.unlink(tmp_path)
        return model

    left_pkl = mano_dir / "MANO_LEFT.pkl"
    right_pkl = mano_dir / "MANO_RIGHT.pkl"
    mano_left = load_resolved_mano(left_pkl, is_rhand=False,
                                   num_pca_comps=args.num_pca_comps, device=args.device)
    mano_left.eval()
    mano_right = load_resolved_mano(right_pkl, is_rhand=True,
                                    num_pca_comps=args.num_pca_comps, device=args.device)
    mano_right.eval()
    print("  MANO models loaded")

    # Find sequences
    data_dir = Path(args.data_dir)
    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if args.split == "train":
        participants = TRAIN_PARTICIPANTS
    elif args.split == "test":
        participants = TEST_PARTICIPANTS
    else:
        participants = TRAIN_PARTICIPANTS | TEST_PARTICIPANTS

    seq_dirs = [d for d in seq_dirs if d.name.split("_")[0] in participants]
    print(f"Found {len(seq_dirs)} sequences for split={args.split}")

    # Process sequences
    db = {}
    for seq_dir in tqdm(seq_dirs, desc="Processing sequences"):
        clips = process_sequence(
            str(seq_dir),
            mano_left,
            mano_right,
            clip_len=args.clip_len,
            overlap=args.overlap,
            include_tips=args.include_tips,
            downsample=args.downsample,
        )
        db.update(clips)

    print(f"\nTotal clips: {len(db)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"hot3d_{args.split}_raw.pt")
    torch.save(db, out_path)
    print(f"Saved to {out_path}")

    # Stats
    if db:
        sample = db[list(db.keys())[0]]
        print(f"Sample shapes:")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
