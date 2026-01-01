# add_kp3d_rot6d_handpca12.py
import copy
import torch
import smplx
from tqdm import tqdm
import os
import numpy as np
import torch
import trimesh
# fingertip vertex ids (same as your make_hand_regressor one-hot vertices)
LEFT_TIP_VERTS  = [5361, 4933, 5058, 5169, 5286]  # index, middle, pinky, ring, thumb
RIGHT_TIP_VERTS = [8079, 7669, 7794, 7905, 8022]  # index, middle, pinky, ring, thumb

@torch.no_grad()
def append_fingertips_from_vertices(out):
    """
    out.vertices: [T, V, 3]
    out.joints:   [T, J, 3]
    return: joints_ext [T, J+10, 3]
    """
    verts = out.vertices  # [T,V,3]
    tipL = verts[:, LEFT_TIP_VERTS, :]   # [T,5,3]
    tipR = verts[:, RIGHT_TIP_VERTS, :]  # [T,5,3]
    joints_ext = torch.cat([out.joints, tipL, tipR], dim=1)
    return joints_ext
    
@torch.no_grad()
def save_one_mesh_obj(out, smplx_layer, frame_idx: int, out_path: str):
    """
    out: smplx_layer(...) の戻り
      - out.vertices: [T, V, 3] (torch)
    smplx_layer.faces: [F, 3] (np or torch)
    """
    verts = out.vertices[frame_idx].detach().cpu().numpy()  # [V,3]
    faces = smplx_layer.faces
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mesh.export(out_path)
    print(f"saved: {out_path}  (frame={frame_idx}, V={verts.shape[0]}, F={faces.shape[0]})")


@torch.no_grad()
def save_all_meshes(out, smplx_layer, out_dir: str, fmt: str = "obj", stride: int = 1):
    """
    各フレームを out_dir/frame_000000.obj のように保存
    stride=2 なら2フレームおき
    """
    os.makedirs(out_dir, exist_ok=True)
    faces = smplx_layer.faces
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    T = out.vertices.shape[0]
    for t in range(0, T, stride):
        verts = out.vertices[t].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        out_path = os.path.join(out_dir, f"frame_{t:06d}.{fmt}")
        mesh.export(out_path)
    print(f"saved {len(range(0,T,stride))} meshes to {out_dir}")
def rotmat_to_axis_angle(rotmat: torch.Tensor):
    """
    rotmat: (..., 3, 3)
    return: (..., 3) axis-angle
    """
    R = rotmat
    cos = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

    angle = torch.acos(cos)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    axis = axis / (2 * torch.sin(angle)[..., None] + 1e-8)
    return axis * angle[..., None]

HUMAN_MODEL_PATH = "./models"  # 環境に合わせて

def rot6d_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
    """
    x6d: [..., 6]  (Zhou et al. rot6d)
    return: [..., 3, 3]
    """
    a1 = x6d[..., 0:3]
    a2 = x6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # [...,3,3]

def get_smplx_layer(device, gender="neutral", hand_pca_comps=12, batch_size=1):
    g = {"neutral": "NEUTRAL", "male": "MALE", "female": "FEMALE"}[gender.lower()]
    layer = smplx.create(
        HUMAN_MODEL_PATH,
        "smplx",
        gender=g,
        use_pca=True,                 # ← hand PCAを使う
        num_pca_comps=hand_pca_comps, # ← 12次元に合わせる
        use_face_contour=True,
        batch_size=batch_size,
    ).to(device)
    layer.eval()
    return layer

@torch.no_grad()
def smpl_params_to_joints_rot6d_handpca(smplx_layer, smpl_params: dict, device):
    # shapes:
    # global_orient [T,6]
    # body_pose     [T,21,6]
    # left/right_hand_pose [T,12] (PCA coeffs)
    # transl [T,3], betas [1,10]
    go6 = smpl_params["global_orient"].to(device)        # [T,6]
    bp6 = smpl_params["body_pose"].to(device)            # [T,21,6]
    lh  = smpl_params["left_hand_pose"].to(device)       # [T,12]
    rh  = smpl_params["right_hand_pose"].to(device)      # [T,12]
    tr  = smpl_params["transl"].to(device)               # [T,3]
    bt  = smpl_params["betas"].to(device)                # [1,10]

    T = go6.shape[0]
    betas = bt.expand(T, -1)                             # [T,10]

    goR = rot6d_to_rotmat(go6)            # [T,1,3,3]
    goR = goR.transpose(-1,-2)
    bpR = rot6d_to_rotmat(bp6)     
    bpR = bpR.transpose(-1,-2)
    goA = rotmat_to_axis_angle(goR)
    bpA = rotmat_to_axis_angle(bpR)


    out = smplx_layer(
        betas=betas,
        global_orient=goA,
        body_pose=bpA,
        left_hand_pose=lh,
        right_hand_pose=rh,
        transl=tr,
        pose2rot=True,
    )
    # save_all_meshes(out, smplx_layer, out_dir="debug/meshes_obj", fmt="obj", stride=1)
    # out.joints: [T, J, 3]
    joint_w_fingertips = append_fingertips_from_vertices(out)
    return joint_w_fingertips

def main(in_pt, out_pt, device="cuda:0", gender="neutral"):
    db = torch.load(in_pt, map_location="cpu", weights_only=False)  # PyTorch2.6対策
    smplx_layer = get_smplx_layer(device, gender=gender, hand_pca_comps=12)

    new_db = copy.deepcopy(db)
    for k in tqdm(list(new_db.keys()), desc="add kp3d"):
        item = new_db[k]
        if "smpl_params" not in item:
            continue
        T = int(item["num_frames"])
        smplx_layer = get_smplx_layer(device, gender=gender, hand_pca_comps=12, batch_size=T)
        joints = smpl_params_to_joints_rot6d_handpca(smplx_layer, item["smpl_params"], device)
        item["kp3d"] = joints.detach().cpu()  # [T,J,3]
        item["num_frames"] = int(item["kp3d"].shape[0])
    torch.save(new_db, out_pt)
    print("saved:", out_pt)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train.pt")
    ap.add_argument("--out_pt", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_new.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gender", default="neutral")
    args = ap.parse_args()
    main(args.in_pt, args.out_pt, device=args.device, gender=args.gender)





# JOINT_NAMES = [
#     'pelvis',
#     'left_hip',
#     'right_hip',
#     'spine1',
#     'left_knee',
#     'right_knee',
#     'spine2',
#     'left_ankle',
#     'right_ankle',
#     'spine3',
#     'left_foot',
#     'right_foot',
#     'neck',
#     'left_collar',
#     'right_collar',
#     'head',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
#     'jaw',

#     'left_eye_smplhf',
#     'right_eye_smplhf',
#     'left_index1',
#     'left_index2',
#     'left_index3',
#     'left_middle1',
#     'left_middle2',
#     'left_middle3',
#     'left_pinky1',
#     'left_pinky2',
#     'left_pinky3',
#     'left_ring1',
#     'left_ring2',
#     'left_ring3',
#     'left_thumb1',
#     'left_thumb2',
#     'left_thumb3',
#     'right_index1',
#     'right_index2',
#     'right_index3',
#     'right_middle1',
#     'right_middle2',
#     'right_middle3',
#     'right_pinky1',
#     'right_pinky2',
#     'right_pinky3',
#     'right_ring1',
#     'right_ring2',
#     'right_ring3',
#     'right_thumb1',
#     'right_thumb2',
#     'right_thumb3',
#     'nose',
#     'right_eye',
#     'left_eye',
#     'right_ear',
#     'left_ear',
#     'left_big_toe',
#     'left_small_toe',
#     'left_heel',
#     'right_big_toe',
#     'right_small_toe',
#     'right_heel',
#     'left_thumb',
#     'left_index',
#     'left_middle',
#     'left_ring',
#     'left_pinky',
#     'right_thumb',
#     'right_index',
#     'right_middle',
#     'right_ring',
#     'right_pinky',]