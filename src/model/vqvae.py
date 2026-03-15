import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EMA Vector Quantizer (same behavior)
# ============================================================
class EMAQuantizer(nn.Module):
    def __init__(self, K: int, D: int, ema_decay: float = 0.99, eps: float = 1e-5, reset_threshold: int = 1):
        super().__init__()
        self.K = K
        self.D = D
        self.ema_decay = ema_decay
        self.eps = eps
        self.reset_threshold = reset_threshold

        embed = torch.randn(K, D) / math.sqrt(D)
        self.register_buffer("codebook", embed)  # [K, D]
        self.register_buffer("ema_cluster_size", torch.zeros(K))     # [K]
        self.register_buffer("ema_codebook_sum", embed.clone())      # [K, D]

    @torch.no_grad()
    def _ema_update(self, onehot_assign: torch.Tensor, x_flat: torch.Tensor):
        cluster_size = onehot_assign.sum(dim=0)          # [K]
        code_sum = onehot_assign.t() @ x_flat            # [K, D]

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_codebook_sum.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps) * n
        new_codebook = self.ema_codebook_sum / smoothed.unsqueeze(1)
        self.codebook.copy_(new_codebook)

    @torch.no_grad()
    def _code_reset(self, x_flat: torch.Tensor):
        dead = self.ema_cluster_size < self.reset_threshold
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return
        idx = torch.randint(0, x_flat.size(0), (num_dead,), device=x_flat.device)
        repl = x_flat[idx]
        self.codebook[dead] = repl
        self.ema_codebook_sum[dead] = repl
        self.ema_cluster_size[dead] = self.reset_threshold

    def forward(self, x: torch.Tensor, do_reset: bool = True):
        # x: [..., D]
        orig_shape = x.shape
        assert orig_shape[-1] == self.D
        x_flat = x.reshape(-1, self.D)

        x2 = (x_flat ** 2).sum(dim=1, keepdim=True)              # [N,1]
        e2 = (self.codebook ** 2).sum(dim=1).unsqueeze(0)        # [1,K]
        xe = x_flat @ self.codebook.t()                          # [N,K]
        dist = x2 + e2 - 2 * xe

        indices = dist.argmin(dim=1)                             # [N]
        x_q = self.codebook[indices].reshape(*orig_shape)
        x_q_st = x + (x_q - x).detach()

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, num_classes=self.K).type_as(x_flat)
                self._ema_update(onehot, x_flat)
                if do_reset:
                    self._code_reset(x_flat)

        return x_q_st, indices.reshape(orig_shape[:-1])


# ============================================================
# Helpers
# ============================================================
def build_1d_sincos_posemb(max_len: int, embed_dim: int = 1024, temperature: float = 10000.0):
    arange = torch.arange(max_len, dtype=torch.float32)  # (N,)
    assert embed_dim % 2 == 0
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out = torch.einsum("n,d->nd", arange, omega)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0)  # (1,N,D)
    return pos_emb


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t[:, None] * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _group_norm(num_channels: int, max_groups: int = 8):
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


def _build_flip_sign(n_joints: int, hand_root: bool = False) -> torch.Tensor:
    """Build sign tensor for x-axis mirror of a single hand.
    Layout per hand (hand_root=False): [RIC(n×3), Rot6D(n×6), Vel(n×3)] = n*12 dims.
    Layout per hand (hand_root=True):  [root(9), RIC(n×3), Rot6D(n×6), Vel(n×3)] = 9 + n*12 dims.
    RIC/Vel: negate x → [-1,+1,+1] per joint.
    Rot6D MRM (M=diag(-1,1,1)): col1→[+1,-1,-1], col2→[-1,+1,+1] per joint.
    Hand root(9): vel [-1,+1,+1] + rot6d [+1,-1,-1,-1,+1,+1].
    """
    parts = []
    if hand_root:
        # hand root: vel(3) + rot6d(6)
        root_vel_sign = torch.tensor([-1, 1, 1], dtype=torch.float32)
        root_rot_sign = torch.tensor([1, -1, -1, -1, 1, 1], dtype=torch.float32)
        parts.append(torch.cat([root_vel_sign, root_rot_sign]))  # [9]
    ric_sign = torch.tensor([-1, 1, 1], dtype=torch.float32).repeat(n_joints)    # n*3
    rot_sign = torch.tensor([1, -1, -1, -1, 1, 1], dtype=torch.float32).repeat(n_joints)  # n*6
    vel_sign = torch.tensor([-1, 1, 1], dtype=torch.float32).repeat(n_joints)    # n*3
    parts.extend([ric_sign, rot_sign, vel_sign])
    return torch.cat(parts)  # [9 + n*12] or [n*12]


# ============================================================
# Rotary Position Embedding (RoPE)
# ============================================================
class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin for RoPE, applied per-head."""
    def __init__(self, head_dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, T, head_dim], cos/sin: [1, 1, T, head_dim]"""
    return x * cos + _rotate_half(x) * sin


# ============================================================
# Transformer blocks (as before; used when enc_type="xformer")
# ============================================================
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act=nn.GELU):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)
        self.act = act()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.fc1(x))
        x = self.act(x)
        x = self.drop(self.fc2(x))
        return x


class Attn(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.attn = Attn(dim, heads=heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class ConvXFormerEncoder1D(nn.Module):
    """
    x: [B,T,Cin] -> [B,T',Cout], T' = T / r
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_frames: int,
        temporal_compress: int,
        use_attn: bool = True,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_pos: bool = True,
        learnable_pos_emb: bool = False,
        post_mlp: bool = True,
    ):
        super().__init__()
        self.r = temporal_compress
        self.use_pos = use_pos

        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
                              kernel_size=self.r, stride=self.r)

        Tp = num_frames // self.r
        if use_pos:
            pos = build_1d_sincos_posemb(Tp, embed_dim=out_dim)
            self.pos = nn.Parameter(pos, requires_grad=learnable_pos_emb)
        else:
            self.pos = None

        if (use_attn is False) or (depth <= 0):
            self.blocks = nn.Identity()
        else:
            self.blocks = nn.Sequential(*[
                Block(out_dim, heads=heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
                for _ in range(depth)
            ])

        if post_mlp:
            self.norm = nn.LayerNorm(out_dim)
            self.post_mlp = Mlp(out_dim, mlp_ratio=mlp_ratio, drop=drop, act=nn.Tanh)
        else:
            self.norm = None
            self.post_mlp = None

    def forward(self, x: torch.Tensor):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,T',Cout]
        if self.pos is not None:
            x = x + self.pos
        x = self.blocks(x)
        if self.post_mlp is not None:
            x = x + self.post_mlp(self.norm(x))
        return x


# ============================================================
# Flow decoder components (DiT-ish, cross-attn cond)
# ============================================================
class FlowCrossAttn(nn.Module):
    """Cross-attention for flow decoder: Q from x, KV from cond."""
    def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, N, C = x.shape
        _, M, _ = cond.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        kv = self.kv(cond).reshape(B, M, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class AdaLN(nn.Module):
    def __init__(self, dim, t_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, dim * 2))

    def forward(self, x, t_emb):
        h = self.norm(x)
        ss = self.to_scale_shift(t_emb)
        scale, shift = ss.chunk(2, dim=-1)
        return h * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, t_dim, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.self_attn = Attn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = FlowCrossAttn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.adaln1 = AdaLN(dim, t_dim)
        self.adaln2 = AdaLN(dim, t_dim)
        self.adaln3 = AdaLN(dim, t_dim)

    def forward(self, x, t_emb, cond, rope=None):
        x = x + self.self_attn(self.adaln1(x, t_emb), rope=rope)
        x = x + self.cross_attn(self.adaln2(x, t_emb), cond)
        x = x + self.mlp(self.adaln3(x, t_emb))
        return x


class FlowDecoder1D(nn.Module):
    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        model_dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        t_dim: int = 512,
        use_x_pos: bool = True,
        max_T: int = 2048,
        use_rope: bool = False,
    ):
        super().__init__()
        self.x_in = nn.Linear(x_dim, model_dim)
        self.cond_in = nn.Linear(cond_dim, model_dim)
        self.use_rope = use_rope

        if use_rope:
            head_dim = model_dim // heads
            self.rope = RotaryEmbedding(head_dim, max_len=max_T)
            self.x_pos = None
        elif use_x_pos:
            pos = build_1d_sincos_posemb(max_T, embed_dim=model_dim)
            self.register_buffer("x_pos", pos, persistent=False)
            self.rope = None
        else:
            self.x_pos = None
            self.rope = None

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self._t_dim = t_dim

        self.blocks = nn.ModuleList([
            DiTBlock(model_dim, heads=heads, t_dim=t_dim, mlp_ratio=mlp_ratio,
                     drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(model_dim)
        self.x_out = nn.Linear(model_dim, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor):
        B, T, _ = x_t.shape
        x = self.x_in(x_t)
        if self.x_pos is not None:
            x = x + self.x_pos[:, :T]
        rope = self.rope(T) if self.rope is not None else None
        cond = self.cond_in(cond_tokens)
        t_emb = timestep_embedding(t, dim=self._t_dim)
        t_emb = self.t_mlp(t_emb)
        for blk in self.blocks:
            x = blk(x, t_emb, cond, rope=rope)
        v = self.x_out(self.out_norm(x))
        return v


# ============================================================
# CNN Encoder (★ここが「容量増やしやすい」版)
#   - width / depth / dilation をいじるだけでパラメータ増減
# ============================================================
class ResConv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, drop: float = 0.0):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
        self.gn1 = _group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
        self.gn2 = _group_norm(channels)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.drop(h)

        return F.gelu(x + h)


class CNNEncoder1D(nn.Module):
    """
    x: [B,T,Cin] -> [B,T',Cout], T' = T / r
    容量アップ: cnn_width↑ / cnn_depth↑ / (optional) dilation_schedule を強くする
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_frames: int,
        temporal_compress: int,
        cnn_width: int = 512,      # ★容量ノブ1
        cnn_depth: int = 8,        # ★容量ノブ2
        cnn_kernel: int = 3,
        dilation_cycle: bool = True,
        dilation_max: int = 8,     # ★容量ノブ3(受容野)
        drop: float = 0.0,
        use_pos: bool = False,     # CNNなら基本False推奨
        learnable_pos_emb: bool = False,
        post_mlp: bool = False,    # CNNなら基本False推奨
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.r = temporal_compress
        self.use_pos = use_pos

        # Downsample along time: [B,Cin,T] -> [B,width,T']
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, cnn_width, kernel_size=self.r, stride=self.r),
            _group_norm(cnn_width),
            nn.GELU(),
        )

        # Residual conv stack
        blocks = []
        for i in range(cnn_depth):
            if dilation_cycle:
                # 1,2,4,...,dilation_max,1,2,...
                d = 2 ** (i % int(math.log2(dilation_max) + 1))
                d = min(d, dilation_max)
            else:
                d = 1
            blocks.append(ResConv1DBlock(cnn_width, kernel=cnn_kernel, dilation=d, drop=drop))
        self.blocks = nn.Sequential(*blocks)

        # Optional positional embedding (not necessary for CNN usually)
        Tp = num_frames // self.r
        if use_pos:
            pos = build_1d_sincos_posemb(Tp, embed_dim=cnn_width)
            self.pos = nn.Parameter(pos, requires_grad=learnable_pos_emb)
        else:
            self.pos = None

        # Project to out_dim
        self.proj = nn.Conv1d(cnn_width, out_dim, kernel_size=1)

        # Optional post-MLP (token-wise)
        if post_mlp:
            self.norm = nn.LayerNorm(out_dim)
            self.post_mlp = Mlp(out_dim, mlp_ratio=mlp_ratio, drop=drop, act=nn.Tanh)
        else:
            self.norm = None
            self.post_mlp = None

    def forward(self, x: torch.Tensor):
        # x: [B,T,Cin]
        x = x.permute(0, 2, 1)          # [B,Cin,T]
        x = self.stem(x)                # [B,width,T']
        x = self.blocks(x)              # [B,width,T']
        if self.pos is not None:
            # pos: [1,T',C] -> [1,C,T']
            pos = self.pos.transpose(1, 2)
            x = x + pos
        x = self.proj(x)                # [B,out_dim,T']
        x = x.permute(0, 2, 1)          # [B,T',out_dim]
        if self.post_mlp is not None:
            x = x + self.post_mlp(self.norm(x))
        return x


# ============================================================
# Decoder (same as yours)
# ============================================================
class Decoder1D(nn.Module):
    def __init__(self, cin, c_hid, cout, up_factor: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(cin, c_hid, 3, padding=1),
            nn.GELU(),
            _group_norm(c_hid),
        )
        self.up_factor = up_factor
        self.post = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1),
            nn.GELU(),
            _group_norm(c_hid),
            nn.Conv1d(c_hid, cout, 3, padding=1),
        )

    def forward(self, x):
        x = self.pre(x)
        if self.up_factor > 1:
            x = F.interpolate(x, scale_factor=self.up_factor, mode="linear", align_corners=False)
        x = self.post(x)
        return x


# ============================================================
# Cross-Attention for dual decoder
# ============================================================
class _CrossAttn(nn.Module):
    """Lightweight cross-attention: Q from x, KV from cond."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cond):
        B, N, C = x.shape
        h = self.heads
        q = self.q(x).reshape(B, N, h, C // h).transpose(1, 2)
        kv = self.kv(cond).reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _DualDecoderBlock(nn.Module):
    """One layer: self-conv + cross-attn + FFN for each branch."""
    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        # body branch
        self.conv_b = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(), _group_norm(dim),
        )
        self.norm_b1 = nn.LayerNorm(dim)
        self.cross_b = _CrossAttn(dim, heads)  # body attends to hand
        self.norm_b2 = nn.LayerNorm(dim)
        self.ffn_b = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim),
        )
        # hand branch
        self.conv_h = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(), _group_norm(dim),
        )
        self.norm_h1 = nn.LayerNorm(dim)
        self.cross_h = _CrossAttn(dim, heads)  # hand attends to body
        self.norm_h2 = nn.LayerNorm(dim)
        self.ffn_h = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, b, h):
        # b, h: [B, T, C]
        # self-conv (channel-first)
        b = b + self.conv_b(b.transpose(1, 2)).transpose(1, 2)
        h = h + self.conv_h(h.transpose(1, 2)).transpose(1, 2)
        # cross-attn
        b = b + self.cross_b(self.norm_b1(b), self.norm_h1(h))
        h = h + self.cross_h(self.norm_h1(h), self.norm_b1(b))
        # FFN
        b = b + self.ffn_b(self.norm_b2(b))
        h = h + self.ffn_h(self.norm_h2(h))
        return b, h


class DualDecoder1D(nn.Module):
    """Separate body/hand decoders with cross-attention interaction."""
    def __init__(self, cin_body, cin_hand, c_hid, cout_body, cout_hand,
                 up_factor=1, depth=4, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.up_factor = up_factor
        # project each branch to shared hidden dim
        self.proj_b = nn.Sequential(
            nn.Conv1d(cin_body, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
        )
        self.proj_h = nn.Sequential(
            nn.Conv1d(cin_hand, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
        )
        # interaction blocks
        self.blocks = nn.ModuleList([
            _DualDecoderBlock(c_hid, heads, mlp_ratio) for _ in range(depth)
        ])
        # output heads
        self.head_b = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
            nn.Conv1d(c_hid, cout_body, 3, padding=1),
        )
        self.head_h = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
            nn.Conv1d(c_hid, cout_hand, 3, padding=1),
        )

    def forward(self, zB, zH):
        """
        zB: [B, cin_body, T']  (channel-first)
        zH: [B, cin_hand, T']  (channel-first)
        returns: [B, cout_body + cout_hand, T]
        """
        b = self.proj_b(zB)  # [B, C, T']
        h = self.proj_h(zH)  # [B, C, T']
        # upsample before interaction
        if self.up_factor > 1:
            b = F.interpolate(b, scale_factor=self.up_factor, mode="linear", align_corners=False)
            h = F.interpolate(h, scale_factor=self.up_factor, mode="linear", align_corners=False)
        # to [B, T, C] for attention
        b = b.transpose(1, 2)
        h = h.transpose(1, 2)
        for blk in self.blocks:
            b, h = blk(b, h)
        # back to channel-first for conv heads
        b = self.head_b(b.transpose(1, 2))
        h = self.head_h(h.transpose(1, 2))
        return torch.cat([b, h], dim=1)  # [B, cout_body+cout_hand, T]


# ============================================================
# Tri-stream Decoder (body / LH / RH with pairwise cross-attn)
# ============================================================
class _TriDecoderBlock(nn.Module):
    """One layer for 3 streams: self-conv + pairwise cross-attn + FFN.
    Each stream has separate cross-attn weights per source (6 pairs total).
    Updates are sequential: body → LH → RH (each sees latest state).
    """
    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        # 6 pairwise cross-attns: {target}_from_{source}
        pairs = [
            ("b", "lh"), ("b", "rh"),     # body ← LH, body ← RH
            ("lh", "b"), ("lh", "rh"),     # LH ← body, LH ← RH
            ("rh", "b"), ("rh", "lh"),     # RH ← body, RH ← LH
        ]
        self.cross = nn.ModuleDict()
        self.cross_norm = nn.ModuleDict()
        for tgt, src in pairs:
            key = f"{tgt}_from_{src}"
            self.cross[key] = _CrossAttn(dim, heads)
            self.cross_norm[f"{key}_q"] = nn.LayerNorm(dim)
            self.cross_norm[f"{key}_kv"] = nn.LayerNorm(dim)

        # per-stream self-conv + FFN
        self.streams = nn.ModuleDict()
        for name in ["b", "lh", "rh"]:
            self.streams[f"conv_{name}"] = nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(), _group_norm(dim),
            )
            self.streams[f"norm_{name}"] = nn.LayerNorm(dim)
            self.streams[f"ffn_{name}"] = nn.Sequential(
                nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim),
            )

    def _update(self, x, sources, name):
        """self-conv → pairwise cross-attn from each source → FFN."""
        # self-conv
        x = x + self.streams[f"conv_{name}"](x.transpose(1, 2)).transpose(1, 2)
        # pairwise cross-attn (separate weights per source)
        for src_name, src_tensor in sources:
            key = f"{name}_from_{src_name}"
            q = self.cross_norm[f"{key}_q"](x)
            kv = self.cross_norm[f"{key}_kv"](src_tensor)
            x = x + self.cross[key](q, kv)
        # FFN
        x = x + self.streams[f"ffn_{name}"](self.streams[f"norm_{name}"](x))
        return x

    def forward(self, b, lh, rh):
        # sequential: body → LH → RH (each sees latest state of prior streams)
        b  = self._update(b,  [("lh", lh), ("rh", rh)], "b")
        lh = self._update(lh, [("b", b),   ("rh", rh)], "lh")
        rh = self._update(rh, [("b", b),   ("lh", lh)], "rh")
        return b, lh, rh


class TriDecoder1D(nn.Module):
    """Separate body/LH/RH decoders with pairwise cross-attention."""
    def __init__(self, cin_body, cin_hand, c_hid, cout_body, cout_hand,
                 up_factor=1, depth=4, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.up_factor = up_factor
        self.proj_b = nn.Sequential(
            nn.Conv1d(cin_body, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
        )
        self.proj_lh = nn.Sequential(
            nn.Conv1d(cin_hand, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
        )
        self.proj_rh = nn.Sequential(
            nn.Conv1d(cin_hand, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
        )
        self.blocks = nn.ModuleList([
            _TriDecoderBlock(c_hid, heads, mlp_ratio) for _ in range(depth)
        ])
        self.head_b = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
            nn.Conv1d(c_hid, cout_body, 3, padding=1),
        )
        self.head_lh = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
            nn.Conv1d(c_hid, cout_hand, 3, padding=1),
        )
        self.head_rh = nn.Sequential(
            nn.Conv1d(c_hid, c_hid, 3, padding=1), nn.GELU(), _group_norm(c_hid),
            nn.Conv1d(c_hid, cout_hand, 3, padding=1),
        )

    def forward(self, zB, zLH, zRH):
        """
        zB:  [B, cin_body, T']
        zLH: [B, cin_hand, T']
        zRH: [B, cin_hand, T']
        returns: (body_out, lh_out, rh_out) each [B, cout_*, T]
        """
        b = self.proj_b(zB)
        lh = self.proj_lh(zLH)
        rh = self.proj_rh(zRH)
        if self.up_factor > 1:
            b = F.interpolate(b, scale_factor=self.up_factor, mode="linear", align_corners=False)
            lh = F.interpolate(lh, scale_factor=self.up_factor, mode="linear", align_corners=False)
            rh = F.interpolate(rh, scale_factor=self.up_factor, mode="linear", align_corners=False)
        b, lh, rh = b.transpose(1, 2), lh.transpose(1, 2), rh.transpose(1, 2)
        for blk in self.blocks:
            b, lh, rh = blk(b, lh, rh)
        b = self.head_b(b.transpose(1, 2))
        lh = self.head_lh(lh.transpose(1, 2))
        rh = self.head_rh(rh.transpose(1, 2))
        return b, lh, rh


# ============================================================
# H2VQ
#   enc_type_H / enc_type_B を "cnn" にすればCNNのみ
#   CNNの容量UPは: cnn_width / cnn_depth / dilation_max を上げるだけ
# ============================================================
class H2VQ(nn.Module):
    def __init__(
        self,
        T: int,
        body_in_dim: int,
        hand_in_dim: int,

        code_dim: int = 512,
        K: int = 512,
        ema_decay: float = 0.99,
        alpha_commit: float = 0.02,

        body_tokens_per_t: int = 2,
        hand_tokens_per_t: int = 4,
        body_down: int = 4,
        hand_down: int = 4,

        # --- encoder choice ---
        enc_type_B: str = "xformer",   # "xformer" or "cnn"
        enc_type_H: str = "xformer",   # "xformer" or "cnn"

        # --- transformer params ---
        enc_use_attn_B: bool = True,
        enc_use_attn_H: bool = True,
        enc_depth: int = 6,
        enc_heads: int = 8,
        mlp_ratio: float = 4.0,
        enc_use_pos: bool = True,
        enc_post_mlp: bool = True,

        # --- CNN params (★容量ノブ) ---
        cnn_width_B: int = 512,
        cnn_depth_B: int = 8,
        cnn_width_H: int = 512,
        cnn_depth_H: int = 8,
        cnn_kernel: int = 3,
        cnn_dilation_max: int = 8,
        cnn_drop: float = 0.0,

        # decoder capacity
        dec_hid: int = 512,
        dec_mode: str = "single",  # "single", "dual", or "tri"
        dec_dual_depth: int = 4,
        dec_dual_heads: int = 4,
        dec_dual_mlp_ratio: float = 2.0,

        # cross-modal masking (dual/tri decoder only)
        mask_prob: float = 0.0,      # prob for each of mask_body / mask_hand / mask_random
        mask_ratio: float = 0.3,     # per-token mask ratio for mask_random mode

        # split hands (LH/RH separate tokenize + LH mirror)
        split_hands: bool = False,

        # loss weights
        alpha_root: float = 1.0,
        alpha_body: float = 1.0,
        alpha_hand: float = 1.0,
        use_root_loss: bool = True,
        include_fingertips: bool = False,

        # world-space joints loss
        alpha_joints: float = 0.0,
        alpha_joints_hand: float = 0.0,
        alpha_bone_length: float = 0.0,
        base_idx: int = 0,
        hand_local: bool = False,

        # --- NEW: hand root (default off = backward compat) ---
        hand_root: bool = False,
        hand_root_dim: int = 9,

        # --- NEW: fusion control (default on = backward compat) ---
        use_fuse: bool = True,

        # --- NEW: token separation (default off = backward compat, 2 codebooks) ---
        use_token_separation: bool = False,
        body_root_tokens_per_t: int = 1,
        body_local_tokens_per_t: int = 3,
        hand_root_tokens_per_t: int = 1,
        hand_local_tokens_per_t: int = 3,

        # --- NEW: three decoders (default off = existing dec_mode) ---
        use_three_decoders: bool = False,
        alpha_body_dec: float = 1.0,
        alpha_hand_dec: float = 1.0,
        alpha_full_dec: float = 1.0,

        # --- NEW: hand trajectory token (separate encoder+codebook for wrist vel+rot6d) ---
        use_hand_traj_token: bool = False,

        # --- NEW: hand-only mode (no body encoder/codebook, hand decoder only) ---
        hand_only: bool = False,

        # --- NEW: decoder type (regressor | flow | diffusion) ---
        decoder_type: str = "regressor",
        # flow/diffusion decoder params
        flow_model_dim: int = 256,
        flow_depth: int = 8,
        flow_heads: int = 8,
        flow_mlp_ratio: float = 4.0,
        flow_drop: float = 0.0,
        flow_attn_drop: float = 0.0,
        flow_t_dim: int = 512,
        flow_use_rope: bool = False,
        flow_cond_type: str = "baseline",  # "baseline" | "decoder_separate"
        lambda_flow: float = 1.0,
        lambda_entropy: float = 1e-3,
        mask_input_dims: bool = True,
        # sampling (eval time)
        flow_sample_steps: int = 30,
        flow_solver: str = "heun",
        # diffusion-specific
        diffusion_timesteps: int = 1000,
        diffusion_schedule: str = "cosine",  # "cosine" | "linear"
    ):
        super().__init__()
        assert enc_type_B in ["xformer", "cnn"]
        assert enc_type_H in ["xformer", "cnn"]
        assert dec_mode in ["single", "dual", "tri"]
        assert decoder_type in ["regressor", "flow", "diffusion"]

        self.T = T
        self.code_dim = code_dim
        self.K = K
        self.alpha_commit = alpha_commit
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.body_tokens_per_t = body_tokens_per_t
        self.hand_tokens_per_t = hand_tokens_per_t
        self.body_down = body_down
        self.hand_down = hand_down
        self.split_hands = split_hands
        self.body_in_dim = body_in_dim
        self.hand_in_dim = hand_in_dim

        self.alpha_root = alpha_root
        self.alpha_body = alpha_body
        self.alpha_hand = alpha_hand
        self.use_root_loss = use_root_loss
        self.include_fingertips = include_fingertips
        self.alpha_joints = alpha_joints
        self.alpha_joints_hand = alpha_joints_hand
        self.alpha_bone_length = alpha_bone_length
        self.base_idx = base_idx
        self.hand_local = hand_local

        # precompute bone pairs for bone length loss
        if alpha_bone_length > 0:
            from src.evaluate.utils import get_bone_pairs
            if include_fingertips:
                from paramUtil_add_tips import t2m_body_hand_kinematic_chain_with_tips as kc
            else:
                from paramUtil import t2m_body_hand_kinematic_chain as kc
            self._bone_pairs = get_bone_pairs(kc)

        # --- NEW flags ---
        self.hand_root = hand_root
        self.hand_root_dim = hand_root_dim
        self.use_fuse = use_fuse
        self.use_token_separation = use_token_separation
        self.use_three_decoders = use_three_decoders
        self.alpha_body_dec = alpha_body_dec
        self.alpha_hand_dec = alpha_hand_dec
        self.alpha_full_dec = alpha_full_dec
        self.use_hand_traj_token = use_hand_traj_token
        self.hand_only = hand_only

        # --- decoder type ---
        self.decoder_type = decoder_type
        self.lambda_flow = lambda_flow
        self.lambda_entropy = lambda_entropy
        self.mask_input_dims = mask_input_dims
        self.flow_cond_type = flow_cond_type
        self.flow_sample_steps = flow_sample_steps
        self.flow_solver = flow_solver

        # Token separation counts
        self.body_root_tokens_per_t = body_root_tokens_per_t
        self.body_local_tokens_per_t = body_local_tokens_per_t
        self.hand_root_tokens_per_t_sep = hand_root_tokens_per_t
        self.hand_local_tokens_per_t_sep = hand_local_tokens_per_t

        # --- split_hands: LH/RH share one encoder + codebook ---
        if split_hands:
            assert hand_tokens_per_t % 2 == 0, "hand_tokens_per_t must be even for split_hands"
            self.single_hand_dim = hand_in_dim // 2
            self.tokens_per_hand = hand_tokens_per_t // 2
            # Compute n_joints for flip sign: depends on whether hand_root is in the dim
            if hand_root:
                n_joints = (self.single_hand_dim - hand_root_dim) // 12
            else:
                n_joints = self.single_hand_dim // 12
            self.register_buffer("_flip_sign", _build_flip_sign(n_joints, hand_root=hand_root))
            enc_hand_in = self.single_hand_dim
            enc_hand_out = self.tokens_per_hand * code_dim
        else:
            self.single_hand_dim = hand_in_dim
            self.tokens_per_hand = hand_tokens_per_t
            enc_hand_in = hand_in_dim
            enc_hand_out = hand_tokens_per_t * code_dim

        # --- hand trajectory token: strip traj from hand encoder input ---
        if use_hand_traj_token:
            assert hand_root, "use_hand_traj_token requires hand_root=True"
            self.hand_traj_dim = hand_root_dim  # 9D per hand
            enc_hand_in = enc_hand_in - hand_root_dim

        hand_out = hand_tokens_per_t * code_dim  # total hand latent dim (both hands)
        body_out = body_tokens_per_t * code_dim

        # ----- Hand Encoder (shared for LH/RH when split_hands) -----
        if enc_type_H == "xformer":
            self.encH = ConvXFormerEncoder1D(
                in_dim=enc_hand_in, out_dim=enc_hand_out,
                num_frames=T, temporal_compress=hand_down,
                use_attn=enc_use_attn_H, depth=enc_depth, heads=enc_heads,
                mlp_ratio=mlp_ratio, use_pos=enc_use_pos, post_mlp=enc_post_mlp,
            )
        else:
            self.encH = CNNEncoder1D(
                in_dim=enc_hand_in, out_dim=enc_hand_out,
                num_frames=T, temporal_compress=hand_down,
                cnn_width=cnn_width_H, cnn_depth=cnn_depth_H,
                cnn_kernel=cnn_kernel, dilation_max=cnn_dilation_max,
                drop=cnn_drop,
                use_pos=False, post_mlp=False,
            )

        # ----- Body Encoder (skip in hand_only mode) -----
        if not hand_only:
            if enc_type_B == "xformer":
                self.encB = ConvXFormerEncoder1D(
                    in_dim=body_in_dim, out_dim=body_out,
                    num_frames=T, temporal_compress=body_down,
                    use_attn=enc_use_attn_B, depth=enc_depth, heads=enc_heads,
                    mlp_ratio=mlp_ratio, use_pos=enc_use_pos, post_mlp=enc_post_mlp,
                )
            else:
                self.encB = CNNEncoder1D(
                    in_dim=body_in_dim, out_dim=body_out,
                    num_frames=T, temporal_compress=body_down,
                    cnn_width=cnn_width_B, cnn_depth=cnn_depth_B,
                    cnn_kernel=cnn_kernel, dilation_max=cnn_dilation_max,
                    drop=cnn_drop,
                    use_pos=False, post_mlp=False,
                )

        # ----- Hand Trajectory Encoder (small, shared for LH/RH) -----
        if use_hand_traj_token:
            self.encHT = CNNEncoder1D(
                in_dim=hand_root_dim,
                out_dim=code_dim,   # 1 token per hand
                num_frames=T, temporal_compress=hand_down,
                cnn_width=64, cnn_depth=2,
                cnn_kernel=3, dilation_max=4,
                drop=cnn_drop,
                use_pos=False, post_mlp=False,
            )

        # ----- Codebooks -----
        if hand_only:
            # hand-only: only hand codebook(s)
            if use_token_separation:
                self.qHR = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
                self.qHL = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
            else:
                self.qH = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
        elif use_token_separation:
            # 4 codebooks: body_root, body_local, hand_root, hand_local
            self.qBR = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
            self.qBL = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
            self.qHR = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
            self.qHL = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
        else:
            # 2 codebooks: body, hand (backward compat)
            self.qH = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
            self.qB = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)

        # Hand trajectory codebook
        if use_hand_traj_token:
            self.qHT = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)

        # ----- Fusion (backward compat: on by default, skip in hand_only) -----
        if use_fuse and not hand_only:
            self.hand_proj = nn.Linear(hand_out, hand_out)
            self.fuse_proj = nn.Linear(hand_out + body_out, body_out)
        else:
            self.hand_proj = None
            self.fuse_proj = None

        # ----- Compute effective total token counts for decoders -----
        hand_traj_tokens_per_hand = 1 if use_hand_traj_token else 0
        if use_token_separation:
            total_body_tokens = body_root_tokens_per_t + body_local_tokens_per_t
            if split_hands:
                total_hand_tokens_per_hand = hand_root_tokens_per_t + hand_local_tokens_per_t + hand_traj_tokens_per_hand
                total_hand_tokens = total_hand_tokens_per_hand * 2
            else:
                total_hand_tokens = hand_root_tokens_per_t + hand_local_tokens_per_t + hand_traj_tokens_per_hand
        else:
            total_body_tokens = body_tokens_per_t
            if split_hands:
                total_hand_tokens = hand_tokens_per_t + hand_traj_tokens_per_hand * 2
            else:
                total_hand_tokens = hand_tokens_per_t + hand_traj_tokens_per_hand

        self._total_body_tokens = total_body_tokens
        self._total_hand_tokens = total_hand_tokens

        # ----- Decoders -----
        self.dec_mode = dec_mode
        self.x_dim = body_in_dim + hand_in_dim

        if decoder_type == "regressor":
            # === Regressor decoder (existing logic, unchanged) ===
            if hand_only:
                if split_hands:
                    per_hand_cin = (total_hand_tokens // 2) * code_dim
                    self.dec_hand = Decoder1D(
                        cin=per_hand_cin, c_hid=dec_hid,
                        cout=self.single_hand_dim, up_factor=hand_down,
                    )
                else:
                    self.dec_hand = Decoder1D(
                        cin=total_hand_tokens * code_dim, c_hid=dec_hid,
                        cout=hand_in_dim, up_factor=hand_down,
                    )
            elif use_three_decoders:
                self.dec_body = Decoder1D(
                    cin=total_body_tokens * code_dim, c_hid=dec_hid,
                    cout=body_in_dim, up_factor=body_down,
                )
                if split_hands:
                    per_hand_cin = (total_hand_tokens // 2) * code_dim
                    self.dec_hand = Decoder1D(
                        cin=per_hand_cin, c_hid=dec_hid,
                        cout=self.single_hand_dim, up_factor=hand_down,
                    )
                else:
                    self.dec_hand = Decoder1D(
                        cin=total_hand_tokens * code_dim, c_hid=dec_hid,
                        cout=hand_in_dim, up_factor=hand_down,
                    )
                if dec_mode == "tri":
                    assert split_hands, "dec_mode='tri' requires split_hands=True"
                    per_hand_cin = (total_hand_tokens // 2) * code_dim
                    self.dec_full = TriDecoder1D(
                        cin_body=total_body_tokens * code_dim,
                        cin_hand=per_hand_cin, c_hid=dec_hid,
                        cout_body=body_in_dim, cout_hand=self.single_hand_dim,
                        up_factor=body_down, depth=dec_dual_depth,
                        heads=dec_dual_heads, mlp_ratio=dec_dual_mlp_ratio,
                    )
                elif dec_mode == "dual":
                    self.dec_full = DualDecoder1D(
                        cin_body=total_body_tokens * code_dim,
                        cin_hand=total_hand_tokens * code_dim,
                        c_hid=dec_hid, cout_body=body_in_dim, cout_hand=hand_in_dim,
                        up_factor=body_down, depth=dec_dual_depth,
                        heads=dec_dual_heads, mlp_ratio=dec_dual_mlp_ratio,
                    )
                else:
                    dec_in_ch = (total_body_tokens + total_hand_tokens) * code_dim
                    self.dec_full = Decoder1D(dec_in_ch, dec_hid, body_in_dim + hand_in_dim, up_factor=body_down)
            else:
                if dec_mode == "dual":
                    self.dec = DualDecoder1D(
                        cin_body=total_body_tokens * code_dim,
                        cin_hand=total_hand_tokens * code_dim,
                        c_hid=dec_hid, cout_body=body_in_dim, cout_hand=hand_in_dim,
                        up_factor=body_down, depth=dec_dual_depth,
                        heads=dec_dual_heads, mlp_ratio=dec_dual_mlp_ratio,
                    )
                elif dec_mode == "tri":
                    assert split_hands, "dec_mode='tri' requires split_hands=True"
                    per_hand_cin = (total_hand_tokens // 2) * code_dim
                    self.dec = TriDecoder1D(
                        cin_body=total_body_tokens * code_dim,
                        cin_hand=per_hand_cin, c_hid=dec_hid,
                        cout_body=body_in_dim, cout_hand=self.single_hand_dim,
                        up_factor=body_down, depth=dec_dual_depth,
                        heads=dec_dual_heads, mlp_ratio=dec_dual_mlp_ratio,
                    )
                else:
                    dec_in_ch = (total_body_tokens + total_hand_tokens) * code_dim
                    self.dec = Decoder1D(dec_in_ch, dec_hid, body_in_dim + hand_in_dim, up_factor=body_down)

        elif decoder_type in ("flow", "diffusion"):
            # === Flow / Diffusion decoder ===
            if flow_cond_type == "decoder_separate":
                # Separate body/hand flow decoders, cross-conditioned
                self.body_cond_marker = nn.Parameter(torch.randn(1, 1, code_dim) * 0.02)
                self.hand_cond_marker = nn.Parameter(torch.randn(1, 1, code_dim) * 0.02)
                max_Tp = T
                cond_temporal_pos = build_1d_sincos_posemb(max_Tp, embed_dim=code_dim)
                self.register_buffer("cond_temporal_pos", cond_temporal_pos.unsqueeze(2), persistent=False)

                self.flow_body = FlowDecoder1D(
                    x_dim=body_in_dim, cond_dim=code_dim,
                    model_dim=flow_model_dim, depth=flow_depth, heads=flow_heads,
                    mlp_ratio=flow_mlp_ratio, drop=flow_drop, attn_drop=flow_attn_drop,
                    t_dim=flow_t_dim, use_x_pos=True, max_T=T, use_rope=flow_use_rope,
                )
                self.flow_hand = FlowDecoder1D(
                    x_dim=hand_in_dim, cond_dim=code_dim,
                    model_dim=flow_model_dim, depth=flow_depth, heads=flow_heads,
                    mlp_ratio=flow_mlp_ratio, drop=flow_drop, attn_drop=flow_attn_drop,
                    t_dim=flow_t_dim, use_x_pos=True, max_T=T, use_rope=flow_use_rope,
                )
            else:
                # Single flow decoder (baseline)
                self.flow = FlowDecoder1D(
                    x_dim=self.x_dim, cond_dim=code_dim,
                    model_dim=flow_model_dim, depth=flow_depth, heads=flow_heads,
                    mlp_ratio=flow_mlp_ratio, drop=flow_drop, attn_drop=flow_attn_drop,
                    t_dim=flow_t_dim, use_x_pos=True, max_T=T, use_rope=flow_use_rope,
                )

            # Body-only / Hand-only flow decoders (three_decoders mode)
            if use_three_decoders:
                self.flow_body_only = FlowDecoder1D(
                    x_dim=body_in_dim, cond_dim=code_dim,
                    model_dim=flow_model_dim, depth=flow_depth, heads=flow_heads,
                    mlp_ratio=flow_mlp_ratio, drop=flow_drop, attn_drop=flow_attn_drop,
                    t_dim=flow_t_dim, use_x_pos=True, max_T=T, use_rope=flow_use_rope,
                )
                self.flow_hand_only = FlowDecoder1D(
                    x_dim=hand_in_dim, cond_dim=code_dim,
                    model_dim=flow_model_dim, depth=flow_depth, heads=flow_heads,
                    mlp_ratio=flow_mlp_ratio, drop=flow_drop, attn_drop=flow_attn_drop,
                    t_dim=flow_t_dim, use_x_pos=True, max_T=T, use_rope=flow_use_rope,
                )

            # Dim weighting mask for flow loss
            w = torch.ones(self.x_dim)
            if use_root_loss:
                w[:4] = alpha_root
            else:
                w[:4] = 0.0
                w[3] = alpha_root
            w[4:body_in_dim] = alpha_body
            w[body_in_dim:] = alpha_hand
            self.register_buffer("dim_weight", w, persistent=False)
            self.register_buffer("dim_keep", (w > 0).float(), persistent=False)

            # Diffusion noise schedule (only used when decoder_type == "diffusion")
            if decoder_type == "diffusion":
                self.diffusion_timesteps = diffusion_timesteps
                if diffusion_schedule == "cosine":
                    # Cosine schedule (Nichol & Dhariwal 2021)
                    s = 0.008
                    steps = torch.arange(diffusion_timesteps + 1, dtype=torch.float64)
                    f = torch.cos((steps / diffusion_timesteps + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
                    alphas_cumprod = f / f[0]
                    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
                    betas = torch.clamp(betas, 0.0, 0.999).float()
                else:  # linear
                    betas = torch.linspace(1e-4, 0.02, diffusion_timesteps).float()
                alphas = 1.0 - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                self.register_buffer("_betas", betas, persistent=False)
                self.register_buffer("_alphas_cumprod", alphas_cumprod, persistent=False)
                self.register_buffer("_sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
                self.register_buffer("_sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)

    def _split_tokens(self, z_seq: torch.Tensor, tokens_per_t: int):
        B, Tp, C = z_seq.shape
        D = self.code_dim
        return z_seq.view(B, Tp, tokens_per_t, D)

    def _merge_tokens(self, z_tok: torch.Tensor):
        B, Tp, tok, D = z_tok.shape
        return z_tok.view(B, Tp, tok * D)
    
    def _split_hands_input(self, mH):
        """Split mH [B,T,hand_in_dim] into (mLH, mRH) each [B,T,single_hand_dim].
        When hand_root=True, layout is [lh_root(9), rh_root(9), ric_lh, ric_rh, rot_lh, rot_rh, vel_lh, vel_rh].
        Each output hand includes its root: [root(9), ric, rot, vel].

        With include_fingertips=True, the hand data order per block is
        [LH_fingers(15), RH_fingers(15), L_tips(5), R_tips(5)].
        We reorder to [LH_fingers+L_tips(20), RH_fingers+R_tips(20)] for correct splitting.
        """
        offset = 0
        if self.hand_root:
            hrd = self.hand_root_dim  # 9
            lh_root = mH[..., :hrd]
            rh_root = mH[..., hrd:2*hrd]
            offset = 2 * hrd
            nj = (self.single_hand_dim - hrd) // 12
        else:
            nj = self.single_hand_dim // 12

        base = offset

        if self.include_fingertips and nj == 20:
            # Data: [LH_f(15), RH_f(15), L_tips(5), R_tips(5)] per block
            nf, nt = 15, 5
            total = 2 * nf + 2 * nt  # 40

            def split_block(start, d):
                lf = mH[..., start:start+nf*d]
                rf = mH[..., start+nf*d:start+2*nf*d]
                lt = mH[..., start+2*nf*d:start+2*nf*d+nt*d]
                rt = mH[..., start+2*nf*d+nt*d:start+total*d]
                return torch.cat([lf, lt], dim=-1), torch.cat([rf, rt], dim=-1)

            lh_ric, rh_ric = split_block(base, 3)
            lh_rot, rh_rot = split_block(base + total * 3, 6)
            lh_vel, rh_vel = split_block(base + total * 3 + total * 6, 3)
            mLH_parts = [lh_ric, lh_rot, lh_vel]
            mRH_parts = [rh_ric, rh_rot, rh_vel]
        else:
            ric = nj * 3
            rot = nj * 6
            vel = nj * 3
            mLH_parts = [mH[..., base:base+ric], mH[..., base+2*ric:base+2*ric+rot], mH[..., base+2*ric+2*rot:base+2*ric+2*rot+vel]]
            mRH_parts = [mH[..., base+ric:base+2*ric], mH[..., base+2*ric+rot:base+2*ric+2*rot], mH[..., base+2*ric+2*rot+vel:]]

        if self.hand_root:
            mLH_parts.insert(0, lh_root)
            mRH_parts.insert(0, rh_root)
        mLH = torch.cat(mLH_parts, dim=-1)
        mRH = torch.cat(mRH_parts, dim=-1)
        return mLH, mRH

    def _flip_hand(self, h):
        """Mirror a single hand via x-axis sign flip. h: [..., single_hand_dim]."""
        return h * self._flip_sign

    def _reassemble_hand_output(self, lh_out, rh_out):
        """Reassemble LH + RH back into hand_in_dim format.
        When hand_root: [lh_root(9), rh_root(9), ric_lh, ric_rh, rot_lh, rot_rh, vel_lh, vel_rh].
        Without: [ric_lh, ric_rh, rot_lh, rot_rh, vel_lh, vel_rh].

        With include_fingertips=True, each hand output is [fingers(15)+tips(5)] = 20 joints.
        We reorder back to [LH_f(15), RH_f(15), L_tips(5), R_tips(5)] per block.
        """
        offset = 0
        parts = []
        if self.hand_root:
            hrd = self.hand_root_dim
            parts.extend([lh_out[..., :hrd], rh_out[..., :hrd]])
            offset = hrd
            nj = (self.single_hand_dim - hrd) // 12
        else:
            nj = self.single_hand_dim // 12

        if self.include_fingertips and nj == 20:
            nf, nt = 15, 5
            ric_d, rot_d = nj * 3, nj * 6

            def unsplit_block(lh_block, rh_block, d):
                """[LH_f+L_tips, RH_f+R_tips] -> [LH_f, RH_f, L_tips, R_tips]"""
                return torch.cat([
                    lh_block[..., :nf*d], rh_block[..., :nf*d],
                    lh_block[..., nf*d:], rh_block[..., nf*d:],
                ], dim=-1)

            lh_ric = lh_out[..., offset:offset+ric_d]
            lh_rot = lh_out[..., offset+ric_d:offset+ric_d+rot_d]
            lh_vel = lh_out[..., offset+ric_d+rot_d:]
            rh_ric = rh_out[..., offset:offset+ric_d]
            rh_rot = rh_out[..., offset+ric_d:offset+ric_d+rot_d]
            rh_vel = rh_out[..., offset+ric_d+rot_d:]

            parts.append(unsplit_block(lh_ric, rh_ric, 3))
            parts.append(unsplit_block(lh_rot, rh_rot, 6))
            parts.append(unsplit_block(lh_vel, rh_vel, 3))
        else:
            ric, rot = nj * 3, nj * 6
            parts.extend([
                lh_out[..., offset:offset+ric], rh_out[..., offset:offset+ric],
                lh_out[..., offset+ric:offset+ric+rot], rh_out[..., offset+ric:offset+ric+rot],
                lh_out[..., offset+ric+rot:], rh_out[..., offset+ric+rot:],
            ])
        return torch.cat(parts, dim=-1)

    def _run_decoder(self, dec, zB_q, zH_q):
        """Run a single decoder (full, or the legacy self.dec) with body+hand tokens.
        Handles tri/dual/single modes and split_hands unflipping.
        """
        if isinstance(dec, TriDecoder1D):
            traj_per_hand = 1 if self.use_hand_traj_token else 0
            per_hand_dim = (self.tokens_per_hand + traj_per_hand) * self.code_dim
            zLH_q = zH_q[..., :per_hand_dim]
            zRH_q = zH_q[..., per_hand_dim:]
            b_out, lh_out, rh_out = dec(
                zB_q.permute(0, 2, 1), zLH_q.permute(0, 2, 1), zRH_q.permute(0, 2, 1))
            lh_out = lh_out.permute(0, 2, 1)
            lh_out = self._flip_hand(lh_out)
            rh_out = rh_out.permute(0, 2, 1)
            b_out = b_out.permute(0, 2, 1)
            hand_out = self._reassemble_hand_output(lh_out, rh_out)
            return torch.cat([b_out, hand_out], dim=-1).contiguous()
        elif isinstance(dec, DualDecoder1D):
            recon = dec(zB_q.permute(0, 2, 1), zH_q.permute(0, 2, 1))
            recon = recon.permute(0, 2, 1)
            if self.split_hands:
                body_part = recon[..., :self.body_in_dim]
                hand_part = recon[..., self.body_in_dim:]
                lh_raw = hand_part[..., :self.single_hand_dim]
                rh_raw = hand_part[..., self.single_hand_dim:]
                lh_flipped = self._flip_hand(lh_raw)
                hand_out = self._reassemble_hand_output(lh_flipped, rh_raw)
                recon = torch.cat([body_part, hand_out], dim=-1)
            return recon.contiguous()
        else:
            # Decoder1D (single)
            z_cat = torch.cat([zB_q, zH_q], dim=-1)
            recon = dec(z_cat.permute(0, 2, 1)).permute(0, 2, 1)
            if self.split_hands:
                body_part = recon[..., :self.body_in_dim]
                hand_part = recon[..., self.body_in_dim:]
                lh_raw = hand_part[..., :self.single_hand_dim]
                rh_raw = hand_part[..., self.single_hand_dim:]
                lh_flipped = self._flip_hand(lh_raw)
                hand_out = self._reassemble_hand_output(lh_flipped, rh_raw)
                recon = torch.cat([body_part, hand_out], dim=-1)
            return recon.contiguous()

    def _decode(self, zB_q, zH_q):
        """Shared decode logic for both forward and decode_from_ids (legacy path)."""
        if self.use_three_decoders:
            return self._run_decoder(self.dec_full, zB_q, zH_q)
        else:
            return self._run_decoder(self.dec, zB_q, zH_q)

    def _decode_body_only(self, zB_q):
        """Body-only decoder. Returns [B,T,body_in_dim]."""
        return self.dec_body(zB_q.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

    def _decode_hand_only(self, zH_q):
        """Hand-only decoder. Returns [B,T,hand_in_dim] (with split_hands handling)."""
        if self.split_hands:
            traj_per_hand = 1 if self.use_hand_traj_token else 0
            per_hand_dim = (self.tokens_per_hand + traj_per_hand) * self.code_dim
            zLH_q = zH_q[..., :per_hand_dim]
            zRH_q = zH_q[..., per_hand_dim:]
            lh_out = self.dec_hand(zLH_q.permute(0, 2, 1)).permute(0, 2, 1)
            rh_out = self.dec_hand(zRH_q.permute(0, 2, 1)).permute(0, 2, 1)
            lh_out = self._flip_hand(lh_out)
            return self._reassemble_hand_output(lh_out, rh_out)
        else:
            return self.dec_hand(zH_q.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

    def _apply_mask(self, zB_q, zH_q):
        """Cross-modal masking (training only, dual decoder only).
        Returns masked (zB_q, zH_q) and a dict of mask fractions for logging.
        """
        B = zB_q.size(0)
        info = {"mask_body_frac": 0.0, "mask_hand_frac": 0.0, "mask_random_frac": 0.0}

        if not self.training or self.mask_prob <= 0 or self.dec_mode not in ("dual", "tri"):
            return zB_q, zH_q, info

        p = self.mask_prob
        # per-sample random draw: [0, p) → mask_body, [p, 2p) → mask_hand,
        # [2p, 3p) → mask_random, [3p, 1) → normal
        rand = torch.rand(B, device=zB_q.device)
        m_body = rand < p
        m_hand = (rand >= p) & (rand < 2 * p)
        m_rand = (rand >= 2 * p) & (rand < 3 * p)

        # mask_body: zero out body tokens for selected samples
        if m_body.any():
            zB_q = zB_q.clone()
            zB_q[m_body] = 0.0

        # mask_hand: zero out hand tokens for selected samples
        if m_hand.any():
            zH_q = zH_q.clone()
            zH_q[m_hand] = 0.0

        # mask_random: per-token random masking on both streams
        if m_rand.any():
            zB_q = zB_q.clone() if not m_body.any() else zB_q  # already cloned if m_body
            zH_q = zH_q.clone() if not m_hand.any() else zH_q

            # split to token level for fine-grained masking
            D = self.code_dim
            n_rand = int(m_rand.sum().item())

            # body tokens: [n_rand, T', body_tok, D]
            zB_rand = zB_q[m_rand].view(n_rand, -1, self.body_tokens_per_t, D)
            tok_mask_B = torch.rand(zB_rand.shape[:3], device=zB_q.device) < self.mask_ratio
            zB_rand[tok_mask_B.unsqueeze(-1).expand_as(zB_rand)] = 0.0
            zB_q[m_rand] = zB_rand.view(n_rand, -1, self.body_tokens_per_t * D)

            # hand tokens: [n_rand, T', hand_tok, D]
            zH_rand = zH_q[m_rand].view(n_rand, -1, self.hand_tokens_per_t, D)
            tok_mask_H = torch.rand(zH_rand.shape[:3], device=zH_q.device) < self.mask_ratio
            zH_rand[tok_mask_H.unsqueeze(-1).expand_as(zH_rand)] = 0.0
            zH_q[m_rand] = zH_rand.view(n_rand, -1, self.hand_tokens_per_t * D)

        info["mask_body_frac"] = float(m_body.sum().item()) / B
        info["mask_hand_frac"] = float(m_hand.sum().item()) / B
        info["mask_random_frac"] = float(m_rand.sum().item()) / B
        return zB_q, zH_q, info

    def decode_from_ids(self, idxH: torch.Tensor = None, idxB: torch.Tensor = None,
                        idxBR: torch.Tensor = None, idxBL: torch.Tensor = None,
                        idxHR: torch.Tensor = None, idxHL: torch.Tensor = None,
                        mode: str = "full"):
        """Decode from codebook indices.
        2-codebook mode: pass idxH, idxB
        4-codebook mode: pass idxBR, idxBL, idxHR, idxHL
        mode: "full", "body_only", "hand_only" (only for use_three_decoders or hand_only)
        """
        if self.hand_only:
            # Hand-only: build zH_q and decode
            if self.use_token_separation:
                assert idxHR is not None and idxHL is not None
                zHR_q = self.qHR.codebook[idxHR]
                zHL_q = self.qHL.codebook[idxHL]
                zH_q = torch.cat([self._merge_tokens(zHR_q), self._merge_tokens(zHL_q)], dim=-1)
            else:
                assert idxH is not None
                zH_q = self._merge_tokens(self.qH.codebook[idxH])
            return self._decode_hand_only(zH_q)

        if self.use_token_separation:
            assert idxBR is not None and idxBL is not None
            zBR_q = self.qBR.codebook[idxBR]
            zBL_q = self.qBL.codebook[idxBL]
            zB_q = torch.cat([self._merge_tokens(zBR_q), self._merge_tokens(zBL_q)], dim=-1)

            if mode != "body_only":
                assert idxHR is not None and idxHL is not None
                zHR_q = self.qHR.codebook[idxHR]
                zHL_q = self.qHL.codebook[idxHL]
                if self.split_hands:
                    # idxHR: (B, T', 2) = [LH_root, RH_root]
                    # idxHL: (B, T', 2) = [LH_local, RH_local]
                    # Must match forward layout: [LH_root, LH_local, RH_root, RH_local]
                    per_hand = idxHR.shape[-1] // 2
                    zLH_root_q = self._merge_tokens(zHR_q[..., :per_hand, :])
                    zRH_root_q = self._merge_tokens(zHR_q[..., per_hand:, :])
                    zLH_local_q = self._merge_tokens(zHL_q[..., :per_hand, :])
                    zRH_local_q = self._merge_tokens(zHL_q[..., per_hand:, :])
                    zH_q = torch.cat([zLH_root_q, zLH_local_q, zRH_root_q, zRH_local_q], dim=-1)
                else:
                    zH_q = torch.cat([self._merge_tokens(zHR_q), self._merge_tokens(zHL_q)], dim=-1)
        else:
            assert idxH is not None and idxB is not None
            zB_q = self._merge_tokens(self.qB.codebook[idxB])
            zH_q = self._merge_tokens(self.qH.codebook[idxH])

        if self.use_three_decoders and mode == "body_only":
            return self._decode_body_only(zB_q)
        elif self.use_three_decoders and mode == "hand_only":
            return self._decode_hand_only(zH_q)
        else:
            return self._decode(zB_q, zH_q)

    # ============================================================
    # Flow / Diffusion methods
    # ============================================================
    def _flow_cond_from_ids(self, indices: dict):
        """Build conditioning tokens for flow decoder from codebook indices.
        Supports both 2-codebook (idxB, idxH) and 4-codebook (idxBR, idxBL, idxHR, idxHL) modes.
        """
        parts = []  # list of (embedding, is_body)

        if "idxBR" in indices and indices["idxBR"] is not None:
            # 4-codebook token_separation mode
            for key, cb in [("idxBR", self.qBR), ("idxBL", self.qBL)]:
                if key in indices and indices[key] is not None:
                    parts.append((cb.codebook[indices[key]], True))
            for key, cb in [("idxHR", self.qHR), ("idxHL", self.qHL)]:
                if key in indices and indices[key] is not None:
                    parts.append((cb.codebook[indices[key]], False))
        else:
            # 2-codebook mode
            idxB = indices.get("idxB")
            idxH = indices.get("idxH")
            if idxB is not None:
                parts.append((self.qB.codebook[idxB], True))
            if idxH is not None:
                parts.append((self.qH.codebook[idxH], False))

        if len(parts) == 0:
            raise ValueError("No codebook indices found in indices dict")

        B, Tp, _, D = parts[0][0].shape

        if self.flow_cond_type == "decoder_separate":
            body_flat, hand_flat = [], []
            for emb, is_body in parts:
                emb = emb + self.cond_temporal_pos[:, :Tp]
                n_tok = emb.shape[2]
                flat = emb.reshape(B, Tp * n_tok, D)
                if is_body:
                    body_flat.append(flat + self.body_cond_marker)
                else:
                    hand_flat.append(flat + self.hand_cond_marker)
            all_flat = body_flat + hand_flat
            return torch.cat(all_flat, dim=1)
        else:
            all_flat = []
            for emb, _ in parts:
                n_tok = emb.shape[2]
                all_flat.append(emb.reshape(B, Tp * n_tok, D))
            return torch.cat(all_flat, dim=1)

    def _flow_cond_body_only(self, indices: dict):
        """Build conditioning from body VQ tokens only."""
        parts = []
        if "idxBR" in indices and indices["idxBR"] is not None:
            for key, cb in [("idxBR", self.qBR), ("idxBL", self.qBL)]:
                if key in indices and indices[key] is not None:
                    parts.append(cb.codebook[indices[key]])
        elif "idxB" in indices and indices["idxB"] is not None:
            parts.append(self.qB.codebook[indices["idxB"]])
        if len(parts) == 0:
            return None
        all_flat = []
        for emb in parts:
            n_tok = emb.shape[2]
            all_flat.append(emb.reshape(emb.shape[0], emb.shape[1] * n_tok, emb.shape[3]))
        return torch.cat(all_flat, dim=1)

    def _flow_cond_hand_only(self, indices: dict):
        """Build conditioning from hand VQ tokens only."""
        parts = []
        if "idxHR" in indices and indices["idxHR"] is not None:
            for key, cb in [("idxHR", self.qHR), ("idxHL", self.qHL)]:
                if key in indices and indices[key] is not None:
                    parts.append(cb.codebook[indices[key]])
        elif "idxH" in indices and indices["idxH"] is not None:
            parts.append(self.qH.codebook[indices["idxH"]])
        if len(parts) == 0:
            return None
        all_flat = []
        for emb in parts:
            n_tok = emb.shape[2]
            all_flat.append(emb.reshape(emb.shape[0], emb.shape[1] * n_tok, emb.shape[3]))
        return torch.cat(all_flat, dim=1)

    def _entropy_term(self, idx: torch.Tensor):
        x = idx.reshape(-1)
        onehot = F.one_hot(x, num_classes=self.K).float()
        p = onehot.mean(dim=0)
        return (p * torch.log(p + 1e-9)).sum()

    def _forward_flow(self, x0, indices: dict, commits: dict):
        """Flow/diffusion forward pass. x0: [B,T,x_dim] target motion."""
        B, T, _ = x0.shape

        # Entropy regularization
        ent = torch.tensor(0.0, device=x0.device)
        for key in ["idxB", "idxH", "idxBR", "idxBL", "idxHR", "idxHL"]:
            if key in indices and indices[key] is not None:
                ent = ent + self._entropy_term(indices[key])
        entropy_loss = self.lambda_entropy * ent

        # Build conditioning
        cond = self._flow_cond_from_ids(indices)

        if self.decoder_type == "diffusion":
            # === DDPM: sample timestep, add noise, predict epsilon ===
            t_int = torch.randint(0, self.diffusion_timesteps, (B,), device=x0.device)
            t_frac = t_int.float() / self.diffusion_timesteps  # [0,1) for DiT timestep emb

            sqrt_ab = self._sqrt_alphas_cumprod[t_int]          # (B,)
            sqrt_1mab = self._sqrt_one_minus_alphas_cumprod[t_int]  # (B,)

            eps = torch.randn_like(x0)
            x_t = sqrt_ab[:, None, None] * x0 + sqrt_1mab[:, None, None] * eps

            if self.mask_input_dims:
                x_t = x_t * self.dim_keep[None, None, :]

            if self.flow_cond_type == "decoder_separate":
                eps_pred_B = self.flow_body(x_t[:, :, :self.body_in_dim], t_frac, cond)
                eps_pred_H = self.flow_hand(x_t[:, :, self.body_in_dim:], t_frac, cond)

                w_B = self.dim_weight[None, None, :self.body_in_dim]
                w_H = self.dim_weight[None, None, self.body_in_dim:]
                gen_loss = self.lambda_flow * (
                    torch.mean(((eps_pred_B - eps[:, :, :self.body_in_dim]) * w_B) ** 2) +
                    torch.mean(((eps_pred_H - eps[:, :, self.body_in_dim:]) * w_H) ** 2)
                )
            else:
                eps_pred = self.flow(x_t, t_frac, cond)
                w = self.dim_weight[None, None, :]
                gen_loss = self.lambda_flow * torch.mean(((eps_pred - eps) * w) ** 2)
        else:
            # === Flow matching: sample t, interpolate, predict velocity ===
            t = torch.rand(B, device=x0.device)
            x1 = torch.randn_like(x0)
            x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
            v_star = x1 - x0

            if self.mask_input_dims:
                x_t = x_t * self.dim_keep[None, None, :]

            if self.flow_cond_type == "decoder_separate":
                v_pred_B = self.flow_body(x_t[:, :, :self.body_in_dim], t, cond)
                v_pred_H = self.flow_hand(x_t[:, :, self.body_in_dim:], t, cond)

                w_B = self.dim_weight[None, None, :self.body_in_dim]
                w_H = self.dim_weight[None, None, self.body_in_dim:]
                gen_loss = self.lambda_flow * (
                    torch.mean(((v_pred_B - v_star[:, :, :self.body_in_dim]) * w_B) ** 2) +
                    torch.mean(((v_pred_H - v_star[:, :, self.body_in_dim:]) * w_H) ** 2)
                )
            else:
                v_pred = self.flow(x_t, t, cond)
                w = self.dim_weight[None, None, :]
                gen_loss = self.lambda_flow * torch.mean(((v_pred - v_star) * w) ** 2)

        commit_loss = commits["commit_loss"]
        loss = gen_loss + commit_loss + entropy_loss

        # Three decoders: body-only + hand-only flow losses
        loss_body_dec = torch.tensor(0.0, device=x0.device)
        loss_hand_dec = torch.tensor(0.0, device=x0.device)
        if self.use_three_decoders and self.training:
            cond_body = self._flow_cond_body_only(indices)
            cond_hand = self._flow_cond_hand_only(indices)
            w_B = self.dim_weight[None, None, :self.body_in_dim]
            w_H = self.dim_weight[None, None, self.body_in_dim:]

            if self.decoder_type == "diffusion":
                x_t_B = x_t[:, :, :self.body_in_dim]
                x_t_H = x_t[:, :, self.body_in_dim:]
                target_B = eps[:, :, :self.body_in_dim]
                target_H = eps[:, :, self.body_in_dim:]
            else:
                x_t_B = x_t[:, :, :self.body_in_dim]
                x_t_H = x_t[:, :, self.body_in_dim:]
                target_B = v_star[:, :, :self.body_in_dim]
                target_H = v_star[:, :, self.body_in_dim:]

            if cond_body is not None:
                pred_B = self.flow_body_only(x_t_B, t_frac if self.decoder_type == "diffusion" else t, cond_body)
                loss_body_dec = self.lambda_flow * torch.mean(((pred_B - target_B) * w_B) ** 2)
            if cond_hand is not None:
                pred_H = self.flow_hand_only(x_t_H, t_frac if self.decoder_type == "diffusion" else t, cond_hand)
                loss_hand_dec = self.lambda_flow * torch.mean(((pred_H - target_H) * w_H) ** 2)

            loss = loss + self.alpha_body_dec * loss_body_dec + self.alpha_hand_dec * loss_hand_dec

        losses = {
            "loss": loss,
            "flow_loss": gen_loss,  # keep key name for train.py logging compat
            "commit_loss": commit_loss,
            "entropy_loss": entropy_loss,
            "recon_loss": gen_loss,  # alias for compat with logging
            "recon_root": torch.tensor(0.0, device=x0.device),
            "recon_body": torch.tensor(0.0, device=x0.device),
            "recon_hand": torch.tensor(0.0, device=x0.device),
        }
        if self.use_three_decoders:
            losses["loss_body_dec"] = loss_body_dec
            losses["loss_hand_dec"] = loss_hand_dec
            losses["loss_full_dec"] = gen_loss
        losses.update(commits)

        return None, losses, indices

    @torch.no_grad()
    def sample_from_ids(self, indices_or_idxH, idxB=None,
                        target_T: int = None, steps: int = None, solver: str = None,
                        mode: str = "full"):
        """ODE sampling from codebook indices (flow/diffusion decoder).
        Accepts either indices dict or (idxH, idxB) for backward compat.
        mode: "full" (default), "body_only", "hand_only"
        """
        if isinstance(indices_or_idxH, dict):
            indices = indices_or_idxH
        else:
            indices = {"idxH": indices_or_idxH, "idxB": idxB}

        if steps is None:
            steps = self.flow_sample_steps
        if solver is None:
            solver = self.flow_solver

        # Get batch size from first available index
        for v in indices.values():
            if v is not None:
                B = v.size(0)
                break

        # Build conditioning based on mode
        if mode == "body_only" and self.use_three_decoders:
            cond = self._flow_cond_body_only(indices)
            dev = cond.device
            x_dim = self.body_in_dim
            decoder = self.flow_body_only
        elif mode == "hand_only" and self.use_three_decoders:
            cond = self._flow_cond_hand_only(indices)
            dev = cond.device
            x_dim = self.hand_in_dim
            decoder = self.flow_hand_only
        else:
            cond = self._flow_cond_from_ids(indices)
            dev = cond.device
            x_dim = self.x_dim
            decoder = None  # use default full decoder logic

        if mode in ("body_only", "hand_only") and self.use_three_decoders:
            # Simple single-decoder sampling
            if self.decoder_type == "diffusion":
                return self._sample_single_ddpm(B, target_T, steps, x_dim, decoder, cond, dev)
            else:
                return self._sample_single_flow(B, target_T, steps, solver, x_dim, decoder, cond, dev)

        if self.decoder_type == "diffusion":
            return self._sample_ddpm(B, target_T, steps, cond, dev)
        else:
            return self._sample_flow(B, target_T, steps, solver, cond, dev)

    def _sample_flow(self, B, target_T, steps, solver, cond, dev):
        """ODE integration for flow matching."""
        if self.flow_cond_type == "decoder_separate":
            x_B = torch.randn(B, target_T, self.body_in_dim, device=dev)
            x_H = torch.randn(B, target_T, self.hand_in_dim, device=dev)
            ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
            for i in range(steps):
                t0, t1 = ts[i].expand(B), ts[i + 1].expand(B)
                dt = t1 - t0
                x_B_in = x_B * self.dim_keep[None, None, :self.body_in_dim] if self.mask_input_dims else x_B
                x_H_in = x_H * self.dim_keep[None, None, self.body_in_dim:] if self.mask_input_dims else x_H
                v0_B = self.flow_body(x_B_in, t0, cond)
                v0_H = self.flow_hand(x_H_in, t0, cond)
                if solver == "euler":
                    x_B = x_B + dt[:, None, None] * v0_B
                    x_H = x_H + dt[:, None, None] * v0_H
                elif solver == "heun":
                    x_B_e = x_B + dt[:, None, None] * v0_B
                    x_H_e = x_H + dt[:, None, None] * v0_H
                    x_B_e_in = x_B_e * self.dim_keep[None, None, :self.body_in_dim] if self.mask_input_dims else x_B_e
                    x_H_e_in = x_H_e * self.dim_keep[None, None, self.body_in_dim:] if self.mask_input_dims else x_H_e
                    v1_B = self.flow_body(x_B_e_in, t1, cond)
                    v1_H = self.flow_hand(x_H_e_in, t1, cond)
                    x_B = x_B + dt[:, None, None] * 0.5 * (v0_B + v1_B)
                    x_H = x_H + dt[:, None, None] * 0.5 * (v0_H + v1_H)
                else:
                    raise ValueError(f"unknown solver: {solver}")
            return torch.cat([x_B, x_H], dim=-1)
        else:
            x = torch.randn(B, target_T, self.x_dim, device=dev)
            ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
            for i in range(steps):
                t0, t1 = ts[i].expand(B), ts[i + 1].expand(B)
                dt = t1 - t0
                x_in = x * self.dim_keep[None, None, :] if self.mask_input_dims else x
                v0 = self.flow(x_in, t0, cond)
                if solver == "euler":
                    x = x + dt[:, None, None] * v0
                elif solver == "heun":
                    x_e = x + dt[:, None, None] * v0
                    x_e_in = x_e * self.dim_keep[None, None, :] if self.mask_input_dims else x_e
                    v1 = self.flow(x_e_in, t1, cond)
                    x = x + dt[:, None, None] * 0.5 * (v0 + v1)
                else:
                    raise ValueError(f"unknown solver: {solver}")
            return x

    def _sample_ddpm(self, B, target_T, steps, cond, dev):
        """DDPM reverse process with optional stride (DDIM-like skip)."""
        total_T = self.diffusion_timesteps
        # Sub-sample timesteps for faster sampling
        stride = max(1, total_T // steps)
        timesteps = list(range(total_T - 1, -1, -stride))
        if timesteps[-1] != 0:
            timesteps.append(0)

        def _predict_eps(x_t, t_frac):
            if self.mask_input_dims:
                x_t = x_t * self.dim_keep[None, None, :]
            if self.flow_cond_type == "decoder_separate":
                e_B = self.flow_body(x_t[:, :, :self.body_in_dim], t_frac, cond)
                e_H = self.flow_hand(x_t[:, :, self.body_in_dim:], t_frac, cond)
                return torch.cat([e_B, e_H], dim=-1)
            else:
                return self.flow(x_t, t_frac, cond)

        x = torch.randn(B, target_T, self.x_dim, device=dev)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, device=dev, dtype=torch.long)
            t_frac = t_batch.float() / total_T

            alpha_bar_t = self._alphas_cumprod[t_cur]
            beta_t = self._betas[t_cur]
            alpha_t = 1.0 - beta_t
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_ab_t = (1.0 - alpha_bar_t).sqrt()

            eps_pred = _predict_eps(x, t_frac)

            # DDPM mean
            x0_pred = (x - sqrt_one_minus_ab_t * eps_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10, 10)  # stability clamp

            if t_cur > 0:
                # Get previous alpha_bar
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_bar_prev = self._alphas_cumprod[t_prev]

                # DDIM deterministic step (eta=0)
                x = (alpha_bar_prev.sqrt() * x0_pred +
                     (1.0 - alpha_bar_prev).sqrt() * eps_pred)
            else:
                x = x0_pred

        return x

    def _sample_single_flow(self, B, target_T, steps, solver, x_dim, decoder, cond, dev):
        """ODE integration for a single flow decoder (body_only / hand_only)."""
        x = torch.randn(B, target_T, x_dim, device=dev)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        for i in range(steps):
            t0, t1 = ts[i].expand(B), ts[i + 1].expand(B)
            dt = t1 - t0
            v0 = decoder(x, t0, cond)
            if solver == "euler":
                x = x + dt[:, None, None] * v0
            elif solver == "heun":
                x_e = x + dt[:, None, None] * v0
                v1 = decoder(x_e, t1, cond)
                x = x + dt[:, None, None] * 0.5 * (v0 + v1)
            else:
                raise ValueError(f"unknown solver: {solver}")
        return x

    def _sample_single_ddpm(self, B, target_T, steps, x_dim, decoder, cond, dev):
        """DDPM reverse process for a single decoder (body_only / hand_only)."""
        total_T = self.diffusion_timesteps
        stride = max(1, total_T // steps)
        timesteps = list(range(total_T - 1, -1, -stride))
        if timesteps[-1] != 0:
            timesteps.append(0)

        x = torch.randn(B, target_T, x_dim, device=dev)
        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, device=dev, dtype=torch.long)
            t_frac = t_batch.float() / total_T
            alpha_bar_t = self._alphas_cumprod[t_cur]
            sqrt_one_minus_ab_t = (1.0 - alpha_bar_t).sqrt()
            eps_pred = decoder(x, t_frac, cond)
            x0_pred = (x - sqrt_one_minus_ab_t * eps_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10, 10)
            if t_cur > 0:
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                alpha_bar_prev = self._alphas_cumprod[t_prev]
                x = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * eps_pred
            else:
                x = x0_pred
        return x

    def _quantize_with_separation(self, zB, zH):
        """Token separation path: split encoder output into root/local, quantize with 4 codebooks."""
        D = self.code_dim

        # Body: split into root tokens and local tokens
        body_root_dim = self.body_root_tokens_per_t * D
        body_local_dim = self.body_local_tokens_per_t * D
        zB_root = zB[..., :body_root_dim]
        zB_local = zB[..., body_root_dim:body_root_dim + body_local_dim]

        zBR_tok = self._split_tokens(zB_root, self.body_root_tokens_per_t)
        zBR_q_tok, idxBR = self.qBR(zBR_tok)
        zBR_q = self._merge_tokens(zBR_q_tok)

        zBL_tok = self._split_tokens(zB_local, self.body_local_tokens_per_t)
        zBL_q_tok, idxBL = self.qBL(zBL_tok)
        zBL_q = self._merge_tokens(zBL_q_tok)

        # Hand: split into root tokens and local tokens (per-hand when split_hands)
        if self.split_hands:
            per_hand_dim = self.tokens_per_hand * D
            hand_root_dim = self.hand_root_tokens_per_t_sep * D
            hand_local_dim = self.hand_local_tokens_per_t_sep * D
            # Each hand's encoder output: [root_tokens, local_tokens]
            zLH = zH[..., :per_hand_dim]
            zRH = zH[..., per_hand_dim:]
            # Split each hand
            zLH_root, zLH_local = zLH[..., :hand_root_dim], zLH[..., hand_root_dim:hand_root_dim+hand_local_dim]
            zRH_root, zRH_local = zRH[..., :hand_root_dim], zRH[..., hand_root_dim:hand_root_dim+hand_local_dim]
            # Shared codebook for LH/RH
            zHR_tok = self._split_tokens(torch.cat([zLH_root, zRH_root], dim=0), self.hand_root_tokens_per_t_sep)
            zHR_q_tok, idxHR_both = self.qHR(zHR_tok)
            zHR_q_both = self._merge_tokens(zHR_q_tok)
            B = zLH.size(0)
            zLH_root_q, zRH_root_q = zHR_q_both[:B], zHR_q_both[B:]
            idxHR_lh, idxHR_rh = idxHR_both[:B], idxHR_both[B:]

            zHL_tok = self._split_tokens(torch.cat([zLH_local, zRH_local], dim=0), self.hand_local_tokens_per_t_sep)
            zHL_q_tok, idxHL_both = self.qHL(zHL_tok)
            zHL_q_both = self._merge_tokens(zHL_q_tok)
            zLH_local_q, zRH_local_q = zHL_q_both[:B], zHL_q_both[B:]
            idxHL_lh, idxHL_rh = idxHL_both[:B], idxHL_both[B:]

            zH_q = torch.cat([zLH_root_q, zLH_local_q, zRH_root_q, zRH_local_q], dim=-1)
            idxHR = torch.cat([idxHR_lh, idxHR_rh], dim=-1)  # for logging
            idxHL = torch.cat([idxHL_lh, idxHL_rh], dim=-1)

            commit_HR = F.mse_loss(
                torch.cat([zLH_root, zRH_root], dim=0),
                zHR_q_both.detach()
            )
            commit_HL = F.mse_loss(
                torch.cat([zLH_local, zRH_local], dim=0),
                zHL_q_both.detach()
            )
        else:
            hand_root_dim = self.hand_root_tokens_per_t_sep * D
            hand_local_dim = self.hand_local_tokens_per_t_sep * D
            zH_root = zH[..., :hand_root_dim]
            zH_local = zH[..., hand_root_dim:hand_root_dim+hand_local_dim]

            zHR_tok = self._split_tokens(zH_root, self.hand_root_tokens_per_t_sep)
            zHR_q_tok, idxHR = self.qHR(zHR_tok)
            zHR_q = self._merge_tokens(zHR_q_tok)

            zHL_tok = self._split_tokens(zH_local, self.hand_local_tokens_per_t_sep)
            zHL_q_tok, idxHL = self.qHL(zHL_tok)
            zHL_q = self._merge_tokens(zHL_q_tok)

            zH_q = torch.cat([zHR_q, zHL_q], dim=-1)
            commit_HR = F.mse_loss(zH_root, zHR_q.detach())
            commit_HL = F.mse_loss(zH_local, zHL_q.detach())

        zB_q = torch.cat([zBR_q, zBL_q], dim=-1)

        commit_BR = F.mse_loss(zB_root, zBR_q.detach())
        commit_BL = F.mse_loss(zB_local, zBL_q.detach())

        commit_loss = self.alpha_commit * (commit_BR + commit_BL + commit_HR + commit_HL)

        indices = {"idxBR": idxBR, "idxBL": idxBL, "idxHR": idxHR, "idxHL": idxHL}
        commits = {"commit_BR": commit_BR, "commit_BL": commit_BL,
                    "commit_HR": commit_HR, "commit_HL": commit_HL,
                    "commit_loss": commit_loss}
        return zB_q, zH_q, indices, commits

    def _compute_recon_loss(self, recon, target):
        """Compute reconstruction loss for a full body+hand reconstruction."""
        body_dim = self.body_in_dim

        recon_root = recon[:, :, :4]
        target_root = target[:, :, :4]
        if self.use_root_loss:
            recon_loss_root = F.mse_loss(recon_root, target_root)
        else:
            recon_loss_root = F.mse_loss(recon[:, :, 3], target[:, :, 3])

        recon_body = recon[:, :, 4:body_dim]
        target_body = target[:, :, 4:body_dim]
        recon_loss_body = F.mse_loss(recon_body, target_body)

        recon_hand = recon[:, :, body_dim:]
        target_hand = target[:, :, body_dim:]
        recon_loss_hand = F.mse_loss(recon_hand, target_hand)

        recon_loss = (
            self.alpha_root * recon_loss_root
            + self.alpha_body * recon_loss_body
            + self.alpha_hand * recon_loss_hand
        )
        return recon_loss, recon_loss_root, recon_loss_body, recon_loss_hand

    def _forward_hand_only(self, mH: torch.Tensor):
        """Hand-only forward: encode hands, quantize, decode with hand-only decoder."""
        B = mH.size(0)

        # ---- encode hands ----
        if self.split_hands:
            mLH, mRH = self._split_hands_input(mH)
            mLH_flip = self._flip_hand(mLH)

            if self.use_hand_traj_token:
                hrd = self.hand_traj_dim
                traj_LH, traj_RH = mLH_flip[..., :hrd], mRH[..., :hrd]
                local_LH, local_RH = mLH_flip[..., hrd:], mRH[..., hrd:]
                traj_both = torch.cat([traj_LH, traj_RH], dim=0)
                zHT = self.encHT(traj_both)
                zLH = self.encH(local_LH)
                zRH = self.encH(local_RH)
            else:
                zHT = None
                zLH = self.encH(mLH_flip)
                zRH = self.encH(mRH)

            zH = torch.cat([zLH, zRH], dim=-1)
        else:
            if self.use_hand_traj_token:
                hrd = self.hand_traj_dim
                traj_H, local_H = mH[..., :hrd], mH[..., hrd:]
                zHT = self.encHT(traj_H)
                zH = self.encH(local_H)
            else:
                zHT = None
                zH = self.encH(mH)

        Tm = zH.size(1)

        # ---- quantize hand trajectory ----
        if self.use_hand_traj_token:
            zHT = zHT[:, :Tm]
            zHT_tok = self._split_tokens(zHT, 1)
            zHT_q_tok, idxHT = self.qHT(zHT_tok)
            zHT_q = self._merge_tokens(zHT_q_tok)
            commit_HT = F.mse_loss(zHT_tok, zHT_q_tok.detach())

        # ---- quantize hand ----
        if self.use_token_separation:
            # Hand token separation only (no body)
            D = self.code_dim
            if self.split_hands:
                per_hand_dim = self.tokens_per_hand * D
                hand_root_dim = self.hand_root_tokens_per_t_sep * D
                hand_local_dim = self.hand_local_tokens_per_t_sep * D
                zLH_enc = zH[..., :per_hand_dim]
                zRH_enc = zH[..., per_hand_dim:]
                zLH_root, zLH_local = zLH_enc[..., :hand_root_dim], zLH_enc[..., hand_root_dim:hand_root_dim+hand_local_dim]
                zRH_root, zRH_local = zRH_enc[..., :hand_root_dim], zRH_enc[..., hand_root_dim:hand_root_dim+hand_local_dim]

                zHR_tok = self._split_tokens(torch.cat([zLH_root, zRH_root], dim=0), self.hand_root_tokens_per_t_sep)
                zHR_q_tok, idxHR_both = self.qHR(zHR_tok)
                zHR_q_both = self._merge_tokens(zHR_q_tok)
                zLH_root_q, zRH_root_q = zHR_q_both[:B], zHR_q_both[B:]
                idxHR_lh, idxHR_rh = idxHR_both[:B], idxHR_both[B:]

                zHL_tok = self._split_tokens(torch.cat([zLH_local, zRH_local], dim=0), self.hand_local_tokens_per_t_sep)
                zHL_q_tok, idxHL_both = self.qHL(zHL_tok)
                zHL_q_both = self._merge_tokens(zHL_q_tok)
                zLH_local_q, zRH_local_q = zHL_q_both[:B], zHL_q_both[B:]
                idxHL_lh, idxHL_rh = idxHL_both[:B], idxHL_both[B:]

                zH_q = torch.cat([zLH_root_q, zLH_local_q, zRH_root_q, zRH_local_q], dim=-1)
                idxHR = torch.cat([idxHR_lh, idxHR_rh], dim=-1)
                idxHL = torch.cat([idxHL_lh, idxHL_rh], dim=-1)

                commit_HR = F.mse_loss(torch.cat([zLH_root, zRH_root], dim=0), zHR_q_both.detach())
                commit_HL = F.mse_loss(torch.cat([zLH_local, zRH_local], dim=0), zHL_q_both.detach())
            else:
                hand_root_dim = self.hand_root_tokens_per_t_sep * D
                hand_local_dim = self.hand_local_tokens_per_t_sep * D
                zH_root = zH[..., :hand_root_dim]
                zH_local = zH[..., hand_root_dim:hand_root_dim+hand_local_dim]

                zHR_tok = self._split_tokens(zH_root, self.hand_root_tokens_per_t_sep)
                zHR_q_tok, idxHR = self.qHR(zHR_tok)
                zHR_q = self._merge_tokens(zHR_q_tok)

                zHL_tok = self._split_tokens(zH_local, self.hand_local_tokens_per_t_sep)
                zHL_q_tok, idxHL = self.qHL(zHL_tok)
                zHL_q = self._merge_tokens(zHL_q_tok)

                zH_q = torch.cat([zHR_q, zHL_q], dim=-1)
                commit_HR = F.mse_loss(zH_root, zHR_q.detach())
                commit_HL = F.mse_loss(zH_local, zHL_q.detach())

            commit_loss = self.alpha_commit * (commit_HR + commit_HL)
            indices = {"idxHR": idxHR, "idxHL": idxHL}
            commits = {"commit_HR": commit_HR, "commit_HL": commit_HL, "commit_loss": commit_loss}
        else:
            # 1 hand codebook
            zH_tok = self._split_tokens(zH, self.hand_tokens_per_t)
            zH_q_tok, idxH = self.qH(zH_tok)
            zH_q = self._merge_tokens(zH_q_tok)

            commit_H = F.mse_loss(zH_tok, zH_q_tok.detach())
            commit_loss = self.alpha_commit * commit_H
            indices = {"idxH": idxH}
            commits = {"commit_H": commit_H, "commit_loss": commit_loss}

        # ---- merge traj tokens ----
        if self.use_hand_traj_token:
            commit_loss = commit_loss + self.alpha_commit * commit_HT
            commits["commit_HT"] = commit_HT
            if self.split_hands:
                zHT_q_LH, zHT_q_RH = zHT_q[:B], zHT_q[B:]
                idxHT_lh, idxHT_rh = idxHT[:B], idxHT[B:]
                per_hand_local = self.tokens_per_hand * self.code_dim
                zH_q = torch.cat([zHT_q_LH, zH_q[..., :per_hand_local],
                                  zHT_q_RH, zH_q[..., per_hand_local:]], dim=-1)
                indices["idxHT"] = torch.cat([idxHT_lh, idxHT_rh], dim=-1)
            else:
                zH_q = torch.cat([zHT_q, zH_q], dim=-1)
                indices["idxHT"] = idxHT

        # ---- flow/diffusion path for hand_only ----
        if self.decoder_type in ("flow", "diffusion"):
            T_enc = zH_q.size(1)
            target_hand = mH[:, :T_enc]
            body_zeros = torch.zeros(B, T_enc, self.body_in_dim, device=mH.device)
            x0 = torch.cat([body_zeros, target_hand], dim=-1)
            return self._forward_flow(x0, indices, commits)

        # ---- decode hand-only (regressor) ----
        recon_hand = self._decode_hand_only(zH_q)
        target_hand = mH[:, :recon_hand.shape[1]]

        # ---- hand recon loss ----
        recon_loss_hand = F.mse_loss(recon_hand, target_hand)
        loss = self.alpha_hand * recon_loss_hand + commit_loss

        # ---- build full recon for compatibility (body=zeros + hand) ----
        T_out = recon_hand.shape[1]
        recon_body = torch.zeros(B, T_out, self.body_in_dim, device=mH.device)
        recon = torch.cat([recon_body, recon_hand], dim=-1)

        losses = {
            "loss": loss,
            "recon_loss": recon_loss_hand,
            "recon_root": torch.tensor(0.0, device=mH.device),
            "recon_body": torch.tensor(0.0, device=mH.device),
            "recon_hand": recon_loss_hand,
            "commit_loss": commit_loss,
        }
        losses.update(commits)

        return recon, losses, indices

    def forward(self, mB: torch.Tensor, mH: torch.Tensor):
        if self.hand_only:
            return self._forward_hand_only(mH)

        # ---- encode hands ----
        if self.split_hands:
            mLH, mRH = self._split_hands_input(mH)
            mLH_flip = self._flip_hand(mLH)

            if self.use_hand_traj_token:
                hrd = self.hand_traj_dim
                # Strip traj (first 9D) from flipped/raw hands
                traj_LH = mLH_flip[..., :hrd]
                traj_RH = mRH[..., :hrd]
                local_LH = mLH_flip[..., hrd:]
                local_RH = mRH[..., hrd:]
                # Encode traj (shared encoder, batch LH+RH)
                traj_both = torch.cat([traj_LH, traj_RH], dim=0)
                zHT = self.encHT(traj_both)  # [2B,T',code_dim]
                # Encode local (shared encoder, batch LH+RH)
                zLH = self.encH(local_LH)
                zRH = self.encH(local_RH)
            else:
                zHT = None
                zLH = self.encH(mLH_flip)  # [B,T', tokens_per_hand*D]
                zRH = self.encH(mRH)       # [B,T', tokens_per_hand*D]

            zH = torch.cat([zLH, zRH], dim=-1)  # [B,T', hand_tok*D]
        else:
            if self.use_hand_traj_token:
                hrd = self.hand_traj_dim
                traj_H = mH[..., :hrd]
                local_H = mH[..., hrd:]
                zHT = self.encHT(traj_H)
                zH = self.encH(local_H)
            else:
                zHT = None
                zH = self.encH(mH)  # [B,T', hand_tok*D]
        zB = self.encB(mB)  # [B,T', body_tok*D]

        Tm = min(zH.size(1), zB.size(1))
        zH, zB = zH[:, :Tm], zB[:, :Tm]

        # ---- quantize hand trajectory (if enabled) ----
        if self.use_hand_traj_token:
            zHT = zHT[:, :Tm]
            zHT_tok = self._split_tokens(zHT, 1)  # [...,1,code_dim]
            zHT_q_tok, idxHT = self.qHT(zHT_tok)
            zHT_q = self._merge_tokens(zHT_q_tok)  # [...,code_dim]
            commit_HT = F.mse_loss(zHT_tok, zHT_q_tok.detach())

        # ---- quantize body & hand local ----
        if self.use_token_separation:
            # Fuse before token separation if enabled
            if self.use_fuse:
                zH_proj = self.hand_proj(zH)
                zB = self.fuse_proj(torch.cat([zH_proj, zB], dim=-1))

            zB_q, zH_q, indices, commits = self._quantize_with_separation(zB, zH)
            commit_loss = commits["commit_loss"]
        else:
            # Legacy 2-codebook path
            zH_tok = self._split_tokens(zH, self.hand_tokens_per_t)
            zH_q_tok, idxH = self.qH(zH_tok)
            zH_q = self._merge_tokens(zH_q_tok)

            if self.use_fuse:
                zH_proj = self.hand_proj(zH_q)
                z_fused = self.fuse_proj(torch.cat([zH_proj, zB], dim=-1))
            else:
                z_fused = zB

            zB_tok = self._split_tokens(z_fused, self.body_tokens_per_t)
            zB_q_tok, idxB = self.qB(zB_tok)
            zB_q = self._merge_tokens(zB_q_tok)

            commit_H = F.mse_loss(zH_tok, zH_q_tok.detach())
            commit_B = F.mse_loss(zB_tok, zB_q_tok.detach())
            commit_loss = self.alpha_commit * (commit_H + commit_B)
            indices = {"idxH": idxH, "idxB": idxB}
            commits = {"commit_H": commit_H, "commit_B": commit_B, "commit_loss": commit_loss}

        # ---- merge traj tokens into zH_q for decoder ----
        if self.use_hand_traj_token:
            commit_loss = commit_loss + self.alpha_commit * commit_HT
            commits["commit_HT"] = commit_HT

            if self.split_hands:
                B_orig = mB.size(0)
                zHT_q_LH, zHT_q_RH = zHT_q[:B_orig], zHT_q[B_orig:]
                idxHT_lh, idxHT_rh = idxHT[:B_orig], idxHT[B_orig:]
                # zH_q layout: [LH_local, RH_local] → [LH_traj, LH_local, RH_traj, RH_local]
                per_hand_local = self.tokens_per_hand * self.code_dim
                zH_q = torch.cat([zHT_q_LH, zH_q[..., :per_hand_local],
                                  zHT_q_RH, zH_q[..., per_hand_local:]], dim=-1)
                indices["idxHT"] = torch.cat([idxHT_lh, idxHT_rh], dim=-1)
            else:
                zH_q = torch.cat([zHT_q, zH_q], dim=-1)
                indices["idxHT"] = idxHT

        target = torch.cat([mB, mH], dim=-1)

        # ---- flow/diffusion decoder branch ----
        if self.decoder_type in ("flow", "diffusion"):
            return self._forward_flow(target, indices, commits)

        # ---- three-decoder path with masking ----
        if self.use_three_decoders and self.training:
            B = zB_q.size(0)
            p = self.mask_prob
            rand = torch.rand(B, device=zB_q.device)

            # Detect hand-only samples (body=zeros, e.g. HOT3D)
            m_handonly = (mB.abs().sum(dim=(1, 2)) == 0)

            m_hand = (rand < p) & ~m_handonly
            m_body = (rand >= p) & (rand < 2 * p) & ~m_handonly

            loss = commit_loss
            losses_extra = {}

            # Hand-only samples (body=zeros): only hand-only decoder
            if m_handonly.any():
                zH_q_ho = zH_q[m_handonly]
                hand_recon = self._decode_hand_only(zH_q_ho)
                target_hand = target[m_handonly, :hand_recon.shape[1], self.body_in_dim:]
                loss_ho = F.mse_loss(hand_recon, target_hand)
                loss = loss + self.alpha_hand_dec * loss_ho
                losses_extra["loss_handonly_samples"] = loss_ho
                losses_extra["handonly_frac"] = float(m_handonly.sum().item()) / B

            if m_hand.any():
                # Hand masked: body-only decoder + full decoder (hand tokens zeroed)
                zB_q_masked = zB_q[m_hand]
                body_recon = self._decode_body_only(zB_q_masked)
                target_body = target[m_hand, :body_recon.shape[1], :self.body_in_dim]
                loss_body_dec = F.mse_loss(body_recon, target_body)

                zH_q_zeroed = torch.zeros_like(zH_q[m_hand])
                full_recon = self._decode(zB_q_masked, zH_q_zeroed)
                target_full = target[m_hand, :full_recon.shape[1]]
                loss_full_masked, _, _, _ = self._compute_recon_loss(full_recon, target_full)

                loss = loss + self.alpha_body_dec * loss_body_dec + self.alpha_full_dec * loss_full_masked
                losses_extra["loss_body_dec"] = loss_body_dec
                losses_extra["loss_full_hand_masked"] = loss_full_masked

            if m_body.any():
                # Body masked: hand-only decoder + full decoder (body tokens zeroed)
                zH_q_masked = zH_q[m_body]
                hand_recon = self._decode_hand_only(zH_q_masked)
                target_hand = target[m_body, :hand_recon.shape[1], self.body_in_dim:]
                loss_hand_dec = F.mse_loss(hand_recon, target_hand)

                zB_q_zeroed = torch.zeros_like(zB_q[m_body])
                full_recon = self._decode(zB_q_zeroed, zH_q_masked)
                target_full = target[m_body, :full_recon.shape[1]]
                loss_full_masked, _, _, _ = self._compute_recon_loss(full_recon, target_full)

                loss = loss + self.alpha_hand_dec * loss_hand_dec + self.alpha_full_dec * loss_full_masked
                losses_extra["loss_hand_dec"] = loss_hand_dec
                losses_extra["loss_full_body_masked"] = loss_full_masked

            # No mask: full decoder (excludes hand-only samples)
            m_none = ~m_hand & ~m_body & ~m_handonly
            if m_none.any():
                recon_full = self._decode(zB_q[m_none], zH_q[m_none])
                target_full = target[m_none, :recon_full.shape[1]]
                loss_full, loss_root, loss_body_part, loss_hand_part = self._compute_recon_loss(recon_full, target_full)
                loss = loss + self.alpha_full_dec * loss_full
                losses_extra["loss_full_dec"] = loss_full

            # For logging, run full decode on all samples (no grad for recon output)
            with torch.no_grad():
                recon = self._decode(zB_q, zH_q)
            target_clipped = target[:, :recon.shape[1]]
            recon_loss, recon_loss_root, recon_loss_body, recon_loss_hand = self._compute_recon_loss(recon, target_clipped)

            losses_extra.update({
                "mask_hand_frac": float(m_hand.sum().item()) / B,
                "mask_body_frac": float(m_body.sum().item()) / B,
            })

        else:
            # ---- legacy path: single/dual decoder with optional masking ----
            zB_q_dec, zH_q_dec, mask_info = self._apply_mask(zB_q, zH_q)
            recon = self._decode(zB_q_dec, zH_q_dec)
            target_clipped = target[:, :recon.shape[1]]
            recon_loss, recon_loss_root, recon_loss_body, recon_loss_hand = self._compute_recon_loss(recon, target_clipped)
            loss = recon_loss + commit_loss
            losses_extra = dict(mask_info)

            # eval-time: also run hand-only / body-only decoders for comparison
            if self.use_three_decoders and not self.training:
                recon_hand_only = self._decode_hand_only(zH_q)
                recon_body_only = self._decode_body_only(zB_q)
                losses_extra["recon_hand_only"] = recon_hand_only
                losses_extra["recon_body_only"] = recon_body_only

        # ---- world-space joints L2 loss / bone length loss ----
        if self.alpha_joints > 0 or self.alpha_joints_hand > 0 or self.alpha_bone_length > 0:
            from src.evaluate.utils import recover_joints_from_body_hand
            joints_num = 62 if self.include_fingertips else 52

            # For three-decoder training path, use the unmasked full-decoder
            # recon (recon_full) which has gradients, instead of the no_grad
            # logging recon.
            if self.use_three_decoders and self.training and m_none.any():
                jl_recon = recon_full
                jl_target = target_full
            else:
                jl_recon = recon
                jl_target = target_clipped

            _hrd = self.hand_root_dim * 2 if self.hand_root else 0
            j_gt = recover_joints_from_body_hand(
                jl_target[..., :self.body_in_dim], jl_target[..., self.body_in_dim:],
                include_fingertips=self.include_fingertips,
                hand_root_dim=_hrd,
                joints_num=joints_num,
                use_root_loss=self.use_root_loss,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
                hand_only=self.hand_only,
            )
            j_pr = recover_joints_from_body_hand(
                jl_recon[..., :self.body_in_dim], jl_recon[..., self.body_in_dim:],
                include_fingertips=self.include_fingertips,
                hand_root_dim=_hrd,
                joints_num=joints_num,
                use_root_loss=self.use_root_loss,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
                hand_only=self.hand_only,
            )
            if self.alpha_joints > 0:
                jl = F.mse_loss(j_pr, j_gt)
                loss = loss + self.alpha_joints * jl
                losses_extra["joints_loss"] = jl
            if self.alpha_joints_hand > 0:
                jl_h = F.mse_loss(j_pr[..., 22:, :], j_gt[..., 22:, :])
                loss = loss + self.alpha_joints_hand * jl_h
                losses_extra["joints_loss_hand"] = jl_h
            if self.alpha_bone_length > 0:
                from src.evaluate.utils import compute_bone_lengths
                bl_gt = compute_bone_lengths(j_gt, self._bone_pairs)
                bl_pr = compute_bone_lengths(j_pr, self._bone_pairs)
                bl_loss = F.mse_loss(bl_pr / bl_gt.detach().clamp(min=1e-4), torch.ones_like(bl_gt))
                loss = loss + self.alpha_bone_length * bl_loss
                losses_extra["bone_length_loss"] = bl_loss

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "recon_root": recon_loss_root,
            "recon_body": recon_loss_body,
            "recon_hand": recon_loss_hand,
            "commit_loss": commit_loss,
        }
        losses.update(commits)
        losses.update(losses_extra)

        return recon, losses, indices


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    # --- CNN版: 容量増やすなら cnn_width_* / cnn_depth_* / cnn_dilation_max を上げる ---
    model = H2VQ(
        T=80,
        body_in_dim=263,
        hand_in_dim=360,

        enc_type_B="cnn",
        enc_type_H="cnn",

        body_down=1,
        hand_down=1,

        # ★容量ノブ
        cnn_width_B=512,
        cnn_depth_B=8,
        cnn_width_H=512,
        cnn_depth_H=8,
        cnn_dilation_max=8,
        cnn_drop=0.0,

        # 量子化/トークン
        code_dim=512,
        body_tokens_per_t=2,
        hand_tokens_per_t=4,

        # decoderも増やしたければ
        dec_hid=512,
    )

    mB = torch.randn(4, 80, 263)
    mH = torch.randn(4, 80, 360)
    recon, losses, idx = model(mB, mH)
    print("recon:", recon.shape)
    print(f"idxH: {idx['idxH'].shape}")
    print(f"idxB: {idx['idxB'].shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    idxH = torch.randint(0, model.K, (4, 80, 4))
    idxB = torch.randint(0, model.K, (4, 80, 2))
    recon = model.decode_from_ids(idxH, idxB)
    print("recon:", recon.shape)
