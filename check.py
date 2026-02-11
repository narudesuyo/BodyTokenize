import torch
import numpy as np
from matplotlib import pyplot as plt
data_dir = "/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_tips.pt"
data = torch.load(data_dir, map_location="cpu", weights_only=False)
data_keys = list(data.keys())
data_item = data["fair_bike_02_10___0___1215"]
# mean = np.load("./preprocess/statistics/motion623_fingertips_mean.npy")
# std = np.load("./preprocess/statistics/motion623_fingertips_std.npy")
kp3d = data_item["kp3d"]
kp52_full = np.concatenate([kp3d[:, :22, :], kp3d[:, 25:55, :], kp3d[:, -10:, :]], axis=1)
print(f"kp52_full shape: {kp52_full.shape}")
exit()



edges = [
    # torso
    (0, 1), (0, 2),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),

    # legs
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),

    # arms
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),

    # left hand (wrist=20), 22..36
    (20, 22), (22, 23), (23, 24), (24, -10),
    (20, 25), (25, 26), (26, 27), (27, -9),
    (20, 28), (28, 29), (29, 30), (30, -8),
    (20, 31), (31, 32), (32, 33), (33, -7),
    (20, 34), (34, 35), (35, 36), (36, -6),

    # right hand (wrist=21), 37..51
    (21, 37), (37, 38), (38, 39), (39, -5),
    (21, 40), (40, 41), (41, 42), (42, -4),
    (21, 43), (43, 44), (44, 45), (45, -3),
    (21, 46), (46, 47), (47, 48), (48, -2),
    (21, 49), (49, 50), (50, 51), (51, -1),
]
        
import numpy as np
import matplotlib.pyplot as plt

def visualize_kp52_full(
    kp52_full,
    edges,
    frame=0,
    save_path="kp52_full.png",
    point_size=30,
    bone_width=2.0,
):
    """
    kp52_full: (T, J, 3) or (J, 3)
    edges: list of (parent, child)
    frame: どのフレームを描くか
    """

    # ---- フレーム取り出し ----
    if kp52_full.ndim == 3:
        kp = kp52_full[frame]   # (J,3)
    else:
        kp = kp52_full          # (J,3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ---- joints ----
    ax.scatter(
        kp[:, 0], kp[:, 1], kp[:, 2],
        s=point_size, c="k"
    )

    # ---- bones ----
    for p, c in edges:
        p_xyz = kp[p]
        c_xyz = kp[c]
        ax.plot(
            [p_xyz[0], c_xyz[0]],
            [p_xyz[1], c_xyz[1]],
            [p_xyz[2], c_xyz[2]],
            c="r",
            linewidth=bone_width,
        )

    # ---- 見た目調整 ----
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 等方スケール（歪み防止）
    mins = kp.min(axis=0)
    maxs = kp.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] saved -> {save_path}")

visualize_kp52_full(
    kp52_full,
    edges=edges,
    frame=0,
    save_path="kp52_full_frame0.png"
)