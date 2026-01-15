import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def visualize_two_motions(
    j_gt,
    j_pr,
    save_path,
    edges=None,
    fps=20,
    rotate=True,
    title="GT vs Pred",
    linewidth=2.0,
    view="all",          # "all"|"body"|"hands"|"lh"|"rh"
    joint_names=None,     # list[str] len=J (optional)
    show_names=False,
    show_indices=True,
    name_stride=1,
    text_offset=(0.0, 0.0, 0.0),
    include_fingertips=False,
    only_gt=False,
):
    """
    j_gt, j_pr: (T, J, 3) world coords (x,y,z), y-up.
    Matplotlib 3Dでは Z が縦に見えるので、描画時に (x,z,y) に入れ替えて
    plotのZ軸を高さ(Y)として使う（= Y-up表示）。

    include_fingertips=True:
      - J を 62 (= 52 + 10) 想定
      - 末尾10個が [L5 tips, R5 tips] で、負index (-10..-1) を edges で参照する
    only_gt=True:
      - pred を一切描画しない（scatter/lines/text/legendもGTのみ）
    """

    # ---------- helper: world(x,y,z) -> plot(x, y_plot, z_plot) ----------
    def to_plot_coords(j):
        return j[..., 0], j[..., 2], j[..., 1]

    # ---------- to numpy ----------
    if torch.is_tensor(j_gt):
        j_gt = j_gt.detach().cpu().numpy()
    if (not only_gt) and torch.is_tensor(j_pr):
        j_pr = j_pr.detach().cpu().numpy()

    assert j_gt.ndim == 3
    T, J, _ = j_gt.shape
    if not only_gt:
        assert j_pr.ndim == 3 and j_pr.shape[:2] == (T, J)

    # ---------- origin align ----------
    j_gt = j_gt - j_gt[0:1, 0:1, :]
    if not only_gt:
        j_pr = j_pr - j_pr[0:1, 0:1, :]

    # ---------- default EDGES ----------
    if edges is None:
        if include_fingertips:
            # fingertips are last 10 joints -> referenced by -10..-1
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

                # left hand (wrist=20), 22..36 + tips (-10..-6)
                (20, 22), (22, 23), (23, 24), (24, 53), # index
                (20, 25), (25, 26), (26, 27), (27, 54), # middle
                (20, 28), (28, 29), (29, 30), (30, 56), # pinky
                (20, 31), (31, 32), (32, 33), (33, 55), # ring
                (20, 34), (34, 35), (35, 36), (36, 52), # thumb

                # right hand (wrist=21), 37..51 + tips (-5..-1)
                (21, 37), (37, 38), (38, 39), (39, 58),
                (21, 40), (40, 41), (41, 42), (42, 59),
                (21, 43), (43, 44), (44, 45), (45, 60),
                (21, 46), (46, 47), (47, 48), (48, 61),
                (21, 49), (49, 50), (50, 51), (51, 57),
            ]
        else:
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
                (20, 22), (22, 23), (23, 24),
                (20, 25), (25, 26), (26, 27),
                (20, 28), (28, 29), (29, 30),
                (20, 31), (31, 32), (32, 33),
                (20, 34), (34, 35), (35, 36),

                # right hand (wrist=21), 37..51
                (21, 37), (37, 38), (38, 39),
                (21, 40), (40, 41), (41, 42),
                (21, 43), (43, 44), (44, 45),
                (21, 46), (46, 47), (47, 48),
                (21, 49), (49, 50), (50, 51),
            ]

    # ---------- view subsets ----------
    BODY_SET = set(range(0, 22))
    WRISTS_SET = {20, 21}
    LEFT_HAND_SET  = set(range(22, 37))
    RIGHT_HAND_SET = set(range(37, 52))

    if include_fingertips:
        # last 10 joints are tips: (J-10 .. J-1)
        TIP_IDXS = set(range(J - 10, J))
        LEFT_TIPS  = set(range(J - 10, J - 5))
        RIGHT_TIPS = set(range(J - 5, J))
        LEFT_HAND_SET  = LEFT_HAND_SET  | LEFT_TIPS
        RIGHT_HAND_SET = RIGHT_HAND_SET | RIGHT_TIPS

    if view == "all":
        keep_joints = None
    elif view == "body":
        keep_joints = BODY_SET | WRISTS_SET
    elif view == "hands":
        keep_joints = LEFT_HAND_SET | RIGHT_HAND_SET | WRISTS_SET
    elif view == "lh":
        keep_joints = LEFT_HAND_SET | {20}
    elif view == "rh":
        keep_joints = RIGHT_HAND_SET | {21}
    else:
        raise ValueError(f"Unknown view={view}")

    # ---------- support negative indices in edges (for fingertips) ----------
    def norm_idx(idx: int) -> int:
        return idx if idx >= 0 else (J + idx)

    edges_norm = [(norm_idx(p), norm_idx(c)) for (p, c) in edges]

    # filter edges + choose keep_list
    if keep_joints is not None:
        keep_list = sorted(list(keep_joints))
        keep_set = set(keep_list)
        edges_v = [(p, c) for (p, c) in edges_norm if (p in keep_set and c in keep_set)]
        idx_map = {orig: i for i, orig in enumerate(keep_list)}
    else:
        keep_list = None
        edges_v = edges_norm
        idx_map = None

    # select joints for scatter/axis range
    if keep_list is not None:
        j_gt_v = j_gt[:, keep_list, :]
        j_pr_v = None if only_gt else j_pr[:, keep_list, :]
    else:
        j_gt_v = j_gt
        j_pr_v = None if only_gt else j_pr

    # ---------- axis range in plot coords ----------
    gx, gy, gz = to_plot_coords(j_gt_v.reshape(-1, 3))
    if only_gt:
        all_xyz = np.stack([gx, gy, gz], axis=1)
    else:
        px, py, pz = to_plot_coords(j_pr_v.reshape(-1, 3))
        all_xyz = np.stack(
            [np.concatenate([gx, px]), np.concatenate([gy, py]), np.concatenate([gz, pz])],
            axis=1
        )

    xyz_min = all_xyz.min(axis=0)
    xyz_max = all_xyz.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    radius = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)

    # ---------- figure ----------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # initial scatter
    xg0, yg0, zg0 = to_plot_coords(j_gt_v[0])
    gt_sc = ax.scatter(xg0, yg0, zg0, s=20, c="blue", label="GT")

    pr_sc = None
    if not only_gt:
        xp0, yp0, zp0 = to_plot_coords(j_pr_v[0])
        pr_sc = ax.scatter(xp0, yp0, zp0, s=20, c="red", label="Pred")

    # bones
    gt_lines, pr_lines = [], []
    for (p, c) in edges_v:
        if keep_list is not None:
            p_i = idx_map[p]
            c_i = idx_map[c]
            jp_g = j_gt_v[0, p_i]
            jc_g = j_gt_v[0, c_i]
        else:
            jp_g = j_gt[0, p]
            jc_g = j_gt[0, c]

        x1, y1, z1 = to_plot_coords(jp_g)
        x2, y2, z2 = to_plot_coords(jc_g)
        lg, = ax.plot([x1, x2], [y1, y2], [z1, z2], c="blue", linewidth=linewidth, alpha=0.9)
        gt_lines.append(lg)

        if not only_gt:
            if keep_list is not None:
                jp_p = j_pr_v[0, p_i]
                jc_p = j_pr_v[0, c_i]
            else:
                jp_p = j_pr[0, p]
                jc_p = j_pr[0, c]

            x1, y1, z1 = to_plot_coords(jp_p)
            x2, y2, z2 = to_plot_coords(jc_p)
            lp, = ax.plot([x1, x2], [y1, y2], [z1, z2], c="red", linewidth=linewidth, alpha=0.9)
            pr_lines.append(lp)

    # optional joint texts
    gt_txts, pr_txts = [], []
    dx, dy, dz = text_offset

    def label_for_joint(orig_idx: int):
        nm = None
        if joint_names is not None and 0 <= orig_idx < len(joint_names):
            nm = joint_names[orig_idx]
        if show_names and show_indices:
            return f"{orig_idx}:{nm}" if nm is not None else str(orig_idx)
        elif show_names:
            return nm if nm is not None else str(orig_idx)
        elif show_indices:
            return str(orig_idx)
        else:
            return None

    if (show_names or show_indices):
        visible_orig = keep_list if keep_list is not None else list(range(J))
        stride = max(1, int(name_stride))
        text_visible_orig = visible_orig[::stride]

        for orig_i in text_visible_orig:
            local_i = idx_map[orig_i] if keep_list is not None else orig_i
            txt = label_for_joint(orig_i)
            if txt is None:
                continue

            x, y, z = to_plot_coords(j_gt_v[0, local_i])
            gt_txts.append(ax.text(x + dx, y + dy, z + dz, txt, fontsize=7, color="blue"))

            if not only_gt:
                x, y, z = to_plot_coords(j_pr_v[0, local_i])
                pr_txts.append(ax.text(x + dx, y + dy, z + dz, txt, fontsize=7, color="red"))
    else:
        text_visible_orig = []

    # axis labels (Y-up => plot Z is world Y)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    if not only_gt:
        ax.legend(loc="upper right")

    ax.view_init(elev=15, azim=45)

    BASE_AZIM = 45
    ROT_DEG_PER_STEP = 2.0

    def update(t):
        xg, yg, zg = to_plot_coords(j_gt_v[t])
        gt_sc._offsets3d = (xg, yg, zg)

        if not only_gt:
            xp, yp, zp = to_plot_coords(j_pr_v[t])
            pr_sc._offsets3d = (xp, yp, zp)

        # update bones
        for k, (p, c) in enumerate(edges_v):
            if keep_list is not None:
                p_i = idx_map[p]; c_i = idx_map[c]
                jp_g = j_gt_v[t, p_i]; jc_g = j_gt_v[t, c_i]
            else:
                jp_g = j_gt[t, p]; jc_g = j_gt[t, c]

            x1, y1, z1 = to_plot_coords(jp_g)
            x2, y2, z2 = to_plot_coords(jc_g)
            gt_lines[k].set_data([x1, x2], [y1, y2])
            gt_lines[k].set_3d_properties([z1, z2])

            if not only_gt:
                if keep_list is not None:
                    jp_p = j_pr_v[t, p_i]; jc_p = j_pr_v[t, c_i]
                else:
                    jp_p = j_pr[t, p]; jc_p = j_pr[t, c]

                x1, y1, z1 = to_plot_coords(jp_p)
                x2, y2, z2 = to_plot_coords(jc_p)
                pr_lines[k].set_data([x1, x2], [y1, y2])
                pr_lines[k].set_3d_properties([z1, z2])

        # update texts
        if (show_names or show_indices) and len(gt_txts) > 0:
            for idx_txt, orig_i in enumerate(text_visible_orig):
                local_i = idx_map[orig_i] if keep_list is not None else orig_i

                x, y, z = to_plot_coords(j_gt_v[t, local_i])
                gt_txts[idx_txt].set_position((x + dx, y + dy))
                gt_txts[idx_txt].set_3d_properties(z + dz)

                if not only_gt:
                    x, y, z = to_plot_coords(j_pr_v[t, local_i])
                    pr_txts[idx_txt].set_position((x + dx, y + dy))
                    pr_txts[idx_txt].set_3d_properties(z + dz)

        if rotate:
            ax.view_init(elev=15, azim=BASE_AZIM + ROT_DEG_PER_STEP * t)

        ax.set_title(f"{title}  frame={t}/{T-1}  view={view}")
        artists = [gt_sc] + gt_lines + gt_txts
        if not only_gt:
            artists += [pr_sc] + pr_lines + pr_txts
        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if save_path.endswith(".mp4"):
        ani.save(save_path, writer="ffmpeg", fps=fps)
    elif save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=fps)
    else:
        raise ValueError("save_path must end with .mp4 or .gif")

    plt.close(fig)