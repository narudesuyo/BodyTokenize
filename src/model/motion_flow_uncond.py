"""
Standalone unconditional motion flow matching model.
Uses same DiT architecture as diffusion model but with flow matching (CFM) training.
Convention: t=0 is clean data (x0), t=1 is noise (x1).
x_t = (1-t)*x0 + t*x1, velocity target v* = x1 - x0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.vqvae import (
    timestep_embedding,
    build_1d_sincos_posemb,
    Attn,
    Mlp,
    AdaLN,
)
from src.model.motion_diffusion import DiTBlockUncond


# ============================================================
# MotionDiTFlow: velocity prediction network (continuous t)
# ============================================================
class MotionDiTFlow(nn.Module):
    """Same architecture as MotionDiT but with continuous t in [0,1] scaled for better conditioning."""

    def __init__(
        self,
        x_dim: int = 761,
        model_dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        t_dim: int = 512,
        max_T: int = 256,
        t_scale: float = 1000.0,
    ):
        super().__init__()
        self.x_in = nn.Linear(x_dim, model_dim)
        pos = build_1d_sincos_posemb(max_T, embed_dim=model_dim)
        self.register_buffer("pos_emb", pos, persistent=False)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self._t_dim = t_dim
        self._t_scale = t_scale

        self.blocks = nn.ModuleList([
            DiTBlockUncond(model_dim, heads=heads, t_dim=t_dim,
                           mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(model_dim)
        self.x_out = nn.Linear(model_dim, x_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, T, x_dim) interpolated motion
        t:   (B,) continuous timesteps in [0, 1]
        returns: (B, T, x_dim) velocity prediction
        """
        B, T, _ = x_t.shape
        x = self.x_in(x_t) + self.pos_emb[:, :T]

        # Scale continuous t to [0, t_scale] for richer Fourier features
        t_emb = timestep_embedding(t * self._t_scale, dim=self._t_dim)
        t_emb = self.t_mlp(t_emb)

        for blk in self.blocks:
            x = blk(x, t_emb)

        return self.x_out(self.out_norm(x))


# ============================================================
# MotionFlowMatching: full wrapper
# ============================================================
class MotionFlowMatching(nn.Module):
    def __init__(
        self,
        x_dim: int = 761,
        body_dim: int = 263,
        model_dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        t_dim: int = 512,
        max_T: int = 256,
        t_scale: float = 1000.0,
        alpha_root: float = 5.0,
        alpha_body: float = 1.0,
        alpha_hand: float = 5.0,
        # joints loss
        joints_loss: bool = False,
        alpha_joints: float = 5.0,
        alpha_joints_hand: float = 5.0,
        alpha_bone_length: float = 0.0,
        include_fingertips: bool = True,
        hand_root: bool = False,
        hand_root_dim: int = 9,
        base_idx: int = 15,
        hand_local: bool = False,
        use_root_loss: bool = False,
        # prediction type & geometric losses
        prediction_type: str = "x0",
        velocity_loss: bool = False,
        alpha_velocity: float = 1.0,
        foot_contact_loss: bool = False,
        alpha_foot_contact: float = 1.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.body_dim = body_dim
        self.alpha_root = alpha_root
        self.alpha_body = alpha_body
        self.alpha_hand = alpha_hand
        self.joints_loss = joints_loss
        self.alpha_joints = alpha_joints
        self.alpha_joints_hand = alpha_joints_hand
        self.alpha_bone_length = alpha_bone_length
        self.include_fingertips = include_fingertips
        self.hand_root = hand_root
        self.hand_root_dim = hand_root_dim
        self.base_idx = base_idx
        self.hand_local = hand_local
        self.use_root_loss = use_root_loss
        self.prediction_type = prediction_type
        self.velocity_loss = velocity_loss
        self.alpha_velocity = alpha_velocity
        self.foot_contact_loss = foot_contact_loss
        self.alpha_foot_contact = alpha_foot_contact
        if alpha_bone_length > 0:
            from src.evaluate.utils import get_bone_pairs
            if include_fingertips:
                from paramUtil_add_tips import t2m_body_hand_kinematic_chain_with_tips as kc
            else:
                from paramUtil import t2m_body_hand_kinematic_chain as kc
            self._bone_pairs = get_bone_pairs(kc)
        self.foot_joint_indices = [7, 10, 8, 11]
        self.foot_contact_dim_start = 259
        self.foot_contact_dim_end = 263

        self.net = MotionDiTFlow(
            x_dim=x_dim,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            t_dim=t_dim,
            max_T=max_T,
            t_scale=t_scale,
        )

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Register normalization stats as buffers for joints loss denormalization."""
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

    def set_target_bone_lengths(self, bone_lengths: torch.Tensor):
        """Register fixed GT bone lengths as buffer (num_bones,)."""
        self.register_buffer("_target_bone_lengths", bone_lengths)

    def forward(self, x0: torch.Tensor):
        """
        Training forward: flow matching loss (x0-pred or velocity-pred).
        x0: (B, T, x_dim) clean motion
        """
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device)
        x1 = torch.randn_like(x0)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * x0 + t_expand * x1

        # Target & x0_pred based on prediction type
        v_star = x1 - x0  # true velocity
        net_out = self.net(x_t, t)

        if self.prediction_type == "x0":
            target = x0
            x0_pred = net_out
        else:  # velocity
            target = v_star
            # x_t = (1-t)*x0 + t*x1  =>  x0 = (x_t - t*v_pred) / (1-t)
            # But simpler: x0 = x_t - t*v_pred (since x_t ≈ x0 + t*v)
            x0_pred = x_t - t_expand * net_out
        x0_pred = torch.clamp(x0_pred, -5, 5)

        # Part-weighted loss
        err = (net_out - target) ** 2

        # Detect hand-only samples (body=zeros, e.g. HOT3D)
        body_is_zero = (x0[:, :, :self.body_dim].abs().sum(dim=(1, 2)) == 0)
        has_body = ~body_is_zero

        hand_loss = err[:, :, self.body_dim:].mean()

        if has_body.any():
            if self.use_root_loss:
                root_loss = err[has_body, :, :4].mean()
            else:
                # Only learn root_y (dim3), skip yaw/vx/vz
                root_loss = err[has_body, :, 3].mean()
            body_loss = err[has_body, :, 4:self.body_dim].mean()
        else:
            root_loss = torch.tensor(0.0, device=x0.device)
            body_loss = torch.tensor(0.0, device=x0.device)

        loss = (self.alpha_root * root_loss +
                self.alpha_body * body_loss +
                self.alpha_hand * hand_loss)

        out = {
            "root_loss": root_loss.detach(),
            "body_loss": body_loss.detach(),
            "hand_loss": hand_loss.detach(),
            "mse": err.mean().detach(),
            "handonly_frac": float(body_is_zero.sum().item()) / B,
        }

        # --- Geometric losses from x0_pred (joints, velocity, foot contact) ---
        need_geo = (self.joints_loss or self.velocity_loss or self.foot_contact_loss or self.alpha_bone_length > 0)
        if need_geo and has_body.any() and hasattr(self, "norm_mean"):
            from src.evaluate.utils import recover_joints_from_body_hand

            x0_gt_jl = x0[has_body]
            x0_pr_jl = x0_pred[has_body]

            x0_gt_dn = x0_gt_jl * self.norm_std + self.norm_mean
            x0_pr_dn = x0_pr_jl * self.norm_std + self.norm_mean

            hand_root_dim_total = self.hand_root_dim * 2 if self.hand_root else 0
            joints_num = 62 if self.include_fingertips else 52

            j_gt = recover_joints_from_body_hand(
                x0_gt_dn[..., :self.body_dim], x0_gt_dn[..., self.body_dim:],
                include_fingertips=self.include_fingertips,
                hand_root_dim=hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=self.use_root_loss,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
                hand_only=getattr(self, "hand_only", False),
            )
            j_pr = recover_joints_from_body_hand(
                x0_pr_dn[..., :self.body_dim], x0_pr_dn[..., self.body_dim:],
                include_fingertips=self.include_fingertips,
                hand_root_dim=hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=self.use_root_loss,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
                hand_only=getattr(self, "hand_only", False),
            )

            # Joints loss
            if self.joints_loss:
                jl = F.mse_loss(j_pr, j_gt)
                loss = loss + self.alpha_joints * jl
                out["joints_loss"] = jl.detach()

                if self.alpha_joints_hand > 0:
                    hand_start = 22
                    jl_h = F.mse_loss(j_pr[..., hand_start:, :], j_gt[..., hand_start:, :])
                    loss = loss + self.alpha_joints_hand * jl_h
                    out["joints_loss_hand"] = jl_h.detach()

            # Bone length loss (fixed GT target)
            if self.alpha_bone_length > 0 and hasattr(self, "_target_bone_lengths"):
                from src.evaluate.utils import compute_bone_lengths
                bl_pr = compute_bone_lengths(j_pr, self._bone_pairs)
                bl_target = self._target_bone_lengths.clamp(min=1e-4)  # (num_bones,)
                bl_loss = F.mse_loss(bl_pr / bl_target, torch.ones_like(bl_pr))
                loss = loss + self.alpha_bone_length * bl_loss
                out["bone_length_loss"] = bl_loss.detach()

            # Velocity loss (temporal smoothness)
            if self.velocity_loss:
                vel_gt = j_gt[:, 1:] - j_gt[:, :-1]
                vel_pr = j_pr[:, 1:] - j_pr[:, :-1]
                vl = F.mse_loss(vel_pr, vel_gt)
                loss = loss + self.alpha_velocity * vl
                out["velocity_loss"] = vl.detach()

            # Foot contact loss
            if self.foot_contact_loss:
                fc_gt = x0_gt_dn[..., self.foot_contact_dim_start:self.foot_contact_dim_end]
                foot_joints_pr = j_pr[:, :, self.foot_joint_indices, :]
                foot_vel = foot_joints_pr[:, 1:] - foot_joints_pr[:, :-1]
                foot_vel_sq = (foot_vel ** 2).sum(dim=-1)
                fc_weight = fc_gt[:, 1:]
                fcl = (fc_weight * foot_vel_sq).mean()
                loss = loss + self.alpha_foot_contact * fcl
                out["foot_contact_loss"] = fcl.detach()

        out["loss"] = loss
        return out

    @torch.no_grad()
    def sample(self, B: int, T: int, steps: int = 30, solver: str = "heun",
               device: torch.device = None):
        """Sample by integrating ODE from t=1 (noise) to t=0 (clean)."""
        if device is None:
            device = next(self.parameters()).device

        x = torch.randn(B, T, self.x_dim, device=device)
        return self._ode_integrate(x, t_start=1.0, steps=steps, solver=solver)

    def _get_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get velocity from net output, handling prediction_type conversion."""
        net_out = self.net(x_t, t)
        if self.prediction_type == "x0":
            t_expand = t[:, None, None].clamp(min=1e-4)
            v = (x_t - net_out) / t_expand
        else:
            v = net_out
        return v

    def _x0_step(self, x: torch.Tensor, t: float, t_next: float, B: int,
                 solver: str = "heun") -> torch.Tensor:
        """
        DDIM-like step for x0 prediction: x_{t'} = ((t-t')/t)*x0 + (t'/t)*x_t.
        Avoids v = (x-x0)/t divergence at small t.
        """
        device = x.device
        t_batch = torch.full((B,), t, device=device)

        x0_1 = self.net(x, t_batch)
        x0_1 = torch.clamp(x0_1, -5, 5)

        if t_next < 1e-6:
            return x0_1

        r = t_next / t  # ratio in (0, 1)
        x_next = (1.0 - r) * x0_1 + r * x

        if solver == "heun":
            # 2nd-order correction: predict x0 again from x_next at t_next
            t_next_batch = torch.full((B,), t_next, device=device)
            x0_2 = self.net(x_next, t_next_batch)
            x0_2 = torch.clamp(x0_2, -5, 5)
            x0_avg = 0.5 * (x0_1 + x0_2)
            x_next = (1.0 - r) * x0_avg + r * x

        return x_next

    @torch.no_grad()
    def _ode_integrate(self, x: torch.Tensor, t_start: float,
                       steps: int = 30, solver: str = "heun"):
        """ODE integration from t_start to 0."""
        B = x.shape[0]
        device = x.device

        if self.prediction_type == "x0":
            # DDIM-like direct x0 stepping (numerically stable)
            ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
            for i in range(steps):
                x = self._x0_step(x, ts[i].item(), ts[i + 1].item(), B, solver)
            return x
        else:
            # Velocity prediction: standard ODE
            ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
            for i in range(steps):
                t0 = ts[i].expand(B)
                t1 = ts[i + 1].expand(B)
                dt = t1 - t0

                v0 = self._get_velocity(x, t0)

                if solver == "euler":
                    x = x + dt[:, None, None] * v0
                elif solver == "heun":
                    x_e = x + dt[:, None, None] * v0
                    v1 = self._get_velocity(x_e, t1)
                    x = x + dt[:, None, None] * 0.5 * (v0 + v1)
                else:
                    raise ValueError(f"unknown solver: {solver}")
            return x

    @torch.no_grad()
    def denoise_from_t(self, x_t: torch.Tensor, t_start: float,
                       steps: int = 30, solver: str = "heun"):
        """Partial ODE integration from t_start to 0 (for evaluation reconstruction)."""
        return self._ode_integrate(x_t, t_start=t_start, steps=steps, solver=solver)
