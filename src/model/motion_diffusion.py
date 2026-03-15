"""
Standalone unconditional motion diffusion model (DDPM with DDIM sampling).
Reuses components from vqvae.py: timestep_embedding, build_1d_sincos_posemb, Attn, Mlp, AdaLN.
"""
import math
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


# ============================================================
# DiT block without cross-attention (unconditional)
# ============================================================
class DiTBlockUncond(nn.Module):
    def __init__(self, dim, heads, t_dim, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.self_attn = Attn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.adaln1 = AdaLN(dim, t_dim)
        self.adaln2 = AdaLN(dim, t_dim)

    def forward(self, x, t_emb):
        x = x + self.self_attn(self.adaln1(x, t_emb))
        x = x + self.mlp(self.adaln2(x, t_emb))
        return x


# ============================================================
# MotionDiT: epsilon-prediction network
# ============================================================
class MotionDiT(nn.Module):
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
    ):
        super().__init__()
        self.x_in = nn.Linear(x_dim, model_dim)
        pos = build_1d_sincos_posemb(max_T, embed_dim=model_dim)
        self.register_buffer("pos_emb", pos, persistent=False)  # (1, max_T, model_dim)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self._t_dim = t_dim

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
        x_t: (B, T, x_dim) noised motion
        t:   (B,) integer timesteps
        returns: (B, T, x_dim) epsilon prediction
        """
        B, T, _ = x_t.shape
        x = self.x_in(x_t) + self.pos_emb[:, :T]

        t_emb = timestep_embedding(t, dim=self._t_dim)
        t_emb = self.t_mlp(t_emb)

        for blk in self.blocks:
            x = blk(x, t_emb)

        return self.x_out(self.out_norm(x))


# ============================================================
# Cosine noise schedule utilities
# ============================================================
def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0, 0.999).float()


# ============================================================
# MotionDiffusion: full DDPM wrapper
# ============================================================
class MotionDiffusion(nn.Module):
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
        diffusion_timesteps: int = 1000,
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
        self.diffusion_timesteps = diffusion_timesteps
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
        self.foot_joint_indices = [7, 10, 8, 11]  # fid_l=(7,10), fid_r=(8,11)
        self.foot_contact_dim_start = 259
        self.foot_contact_dim_end = 263

        self.net = MotionDiT(
            x_dim=x_dim,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            t_dim=t_dim,
            max_T=max_T,
        )

        # Register noise schedule buffers
        betas = cosine_beta_schedule(diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # For posterior q(x_{t-1}|x_t, x_0)
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_log_variance", torch.log(torch.clamp(posterior_var, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Register normalization stats as buffers for joints loss denormalization."""
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

    def set_target_bone_lengths(self, bone_lengths: torch.Tensor):
        """Register fixed GT bone lengths as buffer (num_bones,)."""
        self.register_buffer("_target_bone_lengths", bone_lengths)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """Gather schedule values at timestep t and reshape for broadcasting."""
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def forward(self, x0: torch.Tensor):
        """
        Training forward: compute weighted MSE loss (x0-pred or eps-pred).
        x0: (B, T, x_dim) clean motion
        """
        B = x0.shape[0]
        t = torch.randint(0, self.diffusion_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)

        net_out = self.net(x_t, t)

        # Target & x0_pred based on prediction type
        if self.prediction_type == "x0":
            target = x0
            x0_pred = net_out
        else:  # eps
            target = noise
            sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            x0_pred = (x_t - sqrt_omac * net_out) / sqrt_ac
        x0_pred = torch.clamp(x0_pred, -5, 5)

        # Part-weighted loss
        err = (net_out - target) ** 2  # (B, T, x_dim)

        # Detect hand-only samples (body=zeros, e.g. HOT3D)
        body_is_zero = (x0[:, :, :self.body_dim].abs().sum(dim=(1, 2)) == 0)  # (B,)
        has_body = ~body_is_zero  # (B,)

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

            # Use only body-having samples
            x0_gt_jl = x0[has_body]
            x0_pr_jl = x0_pred[has_body]

            # Denormalize
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
                # GT foot contact labels from body part of denormalized x0
                fc_gt = x0_gt_dn[..., self.foot_contact_dim_start:self.foot_contact_dim_end]  # (B', T, 4)
                # Pred foot joint velocities
                foot_joints_pr = j_pr[:, :, self.foot_joint_indices, :]  # (B', T, 4, 3)
                foot_vel = foot_joints_pr[:, 1:] - foot_joints_pr[:, :-1]  # (B', T-1, 4, 3)
                foot_vel_sq = (foot_vel ** 2).sum(dim=-1)  # (B', T-1, 4)
                # Weight by GT contact: when contact=1, foot should be stationary
                fc_weight = fc_gt[:, 1:]  # align with velocity frames (T-1)
                fcl = (fc_weight * foot_vel_sq).mean()
                loss = loss + self.alpha_foot_contact * fcl
                out["foot_contact_loss"] = fcl.detach()

        out["loss"] = loss
        return out

    @torch.no_grad()
    def sample_ddim(self, B: int, T: int, num_steps: int = 50,
                    eta: float = 0.0, device: torch.device = None):
        """DDIM sampling."""
        if device is None:
            device = self.betas.device

        # Subsequence of timesteps
        step_size = self.diffusion_timesteps // num_steps
        timesteps = list(range(0, self.diffusion_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(B, T, self.x_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            net_out = self.net(x, t_batch)

            ac = self.alphas_cumprod[t_cur]
            if i + 1 < len(timesteps):
                ac_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                ac_prev = torch.tensor(1.0, device=device)

            if self.prediction_type == "x0":
                x0_pred = net_out
                eps = (x - torch.sqrt(ac) * x0_pred) / torch.sqrt(1 - ac)
            else:
                eps = net_out
                x0_pred = (x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)
            x0_pred = torch.clamp(x0_pred, -5, 5)

            sigma = eta * torch.sqrt((1 - ac_prev) / (1 - ac) * (1 - ac / ac_prev))
            dir_xt = torch.sqrt(1 - ac_prev - sigma ** 2) * eps
            noise = torch.randn_like(x) if t_cur > 0 else 0
            x = torch.sqrt(ac_prev) * x0_pred + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def sample_ddpm(self, B: int, T: int, device: torch.device = None):
        """Full DDPM sampling (slow, for reference)."""
        if device is None:
            device = self.betas.device

        x = torch.randn(B, T, self.x_dim, device=device)

        for t_cur in reversed(range(self.diffusion_timesteps)):
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            net_out = self.net(x, t_batch)

            coef1 = self._extract(self.posterior_mean_coef1, t_batch, x.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t_batch, x.shape)

            # x0 prediction
            sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            if self.prediction_type == "x0":
                x0_pred = net_out
            else:
                x0_pred = (x - sqrt_omac * net_out) / sqrt_ac
            x0_pred = torch.clamp(x0_pred, -5, 5)

            mean = coef1 * x0_pred + coef2 * x

            if t_cur > 0:
                log_var = self._extract(self.posterior_log_variance, t_batch, x.shape)
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise
            else:
                x = mean

        return x
