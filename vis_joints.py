import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.dataset.dataloader import MotionDataset
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric


def to_TJ3(kp: torch.Tensor) -> torch.Tensor:
    """
    kp を (T, J, 3) に揃える。
    想定:
      - (T, J, 3) そのまま
      - (J, 3) の1フレーム
      - (T, 3, J) とかは必要ならここに追加
    """
    if kp.ndim == 2 and kp.shape[-1] == 3:
        kp = kp.unsqueeze(0)  # (1, J, 3)
    if kp.ndim != 3 or kp.shape[-1] != 3:
        raise ValueError(f"Unexpected kp shape: {tuple(kp.shape)} (expected (T,J,3) or (J,3))")
    return kp


def save_kp_mp4(kp_TJ3: np.ndarray, save_path: str, fps: int = 20, title: str = ""):
    """
    kp_TJ3: (T, J, 3) numpy
    """
    # ★ ここを追加 ★
    if torch.is_tensor(kp_TJ3):
        kp_TJ3 = kp_TJ3.detach().cpu().numpy()
    T, J, _ = kp_TJ3.shape

    # axis limits (固定してブレないようにする)
    xyz_min = kp_TJ3.reshape(-1, 3).min(axis=0)
    xyz_max = kp_TJ3.reshape(-1, 3).max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    radius = float(np.max(xyz_max - xyz_min) / 2.0) + 1e-6

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # 初期フレーム
    scat = ax.scatter(kp_TJ3[0, :, 0], kp_TJ3[0, :, 1], kp_TJ3[0, :, 2], s=10)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # 見た目を揃える（任意）
    ax.view_init(elev=15, azim=60)

    def update(t):
        pts = kp_TJ3[t]
        # scatter の更新（3Dは少し特殊）
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_title(f"{title}  frame {t+1}/{T}")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2000)
    anim.save(save_path, writer=writer)
    plt.close(fig)


def main():
    dataset = MotionDataset(
        pt_path="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt",
        to_torch=True,
        feet_thre=0.002,
        keys=None,
        kp_field="kp3d",
        assume_y_up=True,
        clip_len=80,
        random_crop=True,
        pad_if_short=True,
    )

    out_dir = "./out_mp4"
    num_samples = min(len(dataset), 50)  # 必要なら増やしてOK

    for i in range(num_samples):
        x = dataset[i]
        path = "/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_new.pt/new_joints/iiith_cooking_135_2___7002___7104.npy"
        j_gt = np.load(path)
        print(f"j_gt shape: {j_gt.shape}")
        kp = x["kp"]  # torch tensor
        kp = torch.from_numpy(j_gt).float()
        print(f"kp shape: {kp.shape}")
        save_path = os.path.join(out_dir, f"sample_{i:02d}/kp.mp4")
        save_kp_mp4(kp, save_path, fps=20, title=f"sample {i:02d}")
        print("saved:", save_path)
        exit()

        body = x["body"]
        hand = x["hand"]
        print(body.shape, hand.shape)
        motion = torch.cat([body, hand], dim=-1)
        motion_623 = reconstruct_623_from_body_hand(body.unsqueeze(0), hand.unsqueeze(0))
        j_gt = recover_from_ric(motion_623, joints_num=52)
        j_gt = j_gt.detach().cpu().float().numpy()
        print(f"j_gt shape: {j_gt.shape}")
        save_path = os.path.join(out_dir, f"sample_{i:02d}/623.mp4")
        save_kp_mp4(j_gt[0], save_path, fps=20, title=f"sample {i:02d}")
        print("saved:", save_path)


if __name__ == "__main__":
    main()