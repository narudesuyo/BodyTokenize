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


def _group_norm(num_channels: int, max_groups: int = 8):
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
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

        # loss weights
        alpha_root: float = 1.0,
        alpha_body: float = 1.0,
        alpha_hand: float = 1.0,
        use_root_loss: bool = True,
        include_fingertips: bool = False,
    ):
        super().__init__()
        assert enc_type_B in ["xformer", "cnn"]
        assert enc_type_H in ["xformer", "cnn"]

        self.T = T
        self.code_dim = code_dim
        self.K = K
        self.alpha_commit = alpha_commit
        self.body_tokens_per_t = body_tokens_per_t
        self.hand_tokens_per_t = hand_tokens_per_t
        self.body_down = body_down
        self.hand_down = hand_down

        self.alpha_root = alpha_root
        self.alpha_body = alpha_body
        self.alpha_hand = alpha_hand
        self.use_root_loss = use_root_loss
        self.include_fingertips = include_fingertips
        hand_out = hand_tokens_per_t * code_dim
        body_out = body_tokens_per_t * code_dim

        # ----- Hand Encoder -----
        if enc_type_H == "xformer":
            self.encH = ConvXFormerEncoder1D(
                in_dim=hand_in_dim, out_dim=hand_out,
                num_frames=T, temporal_compress=hand_down,
                use_attn=enc_use_attn_H, depth=enc_depth, heads=enc_heads,
                mlp_ratio=mlp_ratio, use_pos=enc_use_pos, post_mlp=enc_post_mlp,
            )
        else:
            self.encH = CNNEncoder1D(
                in_dim=hand_in_dim, out_dim=hand_out,
                num_frames=T, temporal_compress=hand_down,
                cnn_width=cnn_width_H, cnn_depth=cnn_depth_H,
                cnn_kernel=cnn_kernel, dilation_max=cnn_dilation_max,
                drop=cnn_drop,
                use_pos=False, post_mlp=False,
            )

        # ----- Body Encoder -----
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

        self.qH = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
        self.qB = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)

        self.hand_proj = nn.Linear(hand_out, hand_out)
        self.fuse_proj = nn.Linear(hand_out + body_out, body_out)

        dec_in_ch = (body_tokens_per_t + hand_tokens_per_t) * code_dim
        self.dec = Decoder1D(dec_in_ch, dec_hid, body_in_dim + hand_in_dim, up_factor=body_down)

    def _split_tokens(self, z_seq: torch.Tensor, tokens_per_t: int):
        B, Tp, C = z_seq.shape
        D = self.code_dim
        return z_seq.view(B, Tp, tokens_per_t, D)

    def _merge_tokens(self, z_tok: torch.Tensor):
        B, Tp, tok, D = z_tok.shape
        return z_tok.view(B, Tp, tok * D)
    
    def decode_from_ids(self, idxH: torch.Tensor, idxB: torch.Tensor):
        zH_q_tok = self.qH.codebook[idxH]
        zB_q_tok = self.qB.codebook[idxB]

        zH_q = self._merge_tokens(zH_q_tok)
        zB_q = self._merge_tokens(zB_q_tok)

        z_cat = torch.cat([zB_q, zH_q], dim=-1)
        recon = self.dec(z_cat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        return recon





    def forward(self, mB: torch.Tensor, mH: torch.Tensor):
        zH = self.encH(mH)  # [B,T', hand_tok*D]
        zB = self.encB(mB)  # [B,T', body_tok*D]

        Tm = min(zH.size(1), zB.size(1))
        zH, zB = zH[:, :Tm], zB[:, :Tm]

        # ---- hand quantize first ----
        zH_tok = self._split_tokens(zH, self.hand_tokens_per_t)
        zH_q_tok, idxH = self.qH(zH_tok)
        zH_q = self._merge_tokens(zH_q_tok)

        # ---- fuse to body stream then quantize body ----
        zH_proj = self.hand_proj(zH_q)
        z_fused = self.fuse_proj(torch.cat([zH_proj, zB], dim=-1))

        zB_tok = self._split_tokens(z_fused, self.body_tokens_per_t)
        zB_q_tok, idxB = self.qB(zB_tok)
        zB_q = self._merge_tokens(zB_q_tok)

        # ---- decode ----
        z_cat = torch.cat([zB_q, zH_q], dim=-1)                # [B,T', (bt+ht)*D]
        recon = self.dec(z_cat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        target = torch.cat([mB, mH], dim=-1)[:, :recon.shape[1]]

        recon_root = recon[:, :, :4]
        target_root = target[:, :, :4]
        if self.use_root_loss:
            recon_loss_root = F.mse_loss(recon_root, target_root)
        else:
            recon_loss_root = F.mse_loss(recon[:,:,3:4], target[:,:,3:4])

        recon_body = recon[:, :, 4:263-4]
        target_body = target[:, :, 4:263-4]
        recon_loss_body = F.mse_loss(recon_body, target_body)

        recon_hand = recon[:, :, 263:]
        target_hand = target[:, :, 263:]
        recon_loss_hand = F.mse_loss(recon_hand, target_hand)

        commit_H = F.mse_loss(zH_tok, zH_q_tok.detach())
        commit_B = F.mse_loss(zB_tok, zB_q_tok.detach())
        commit_loss = self.alpha_commit * (commit_H + commit_B)

        recon_loss = (
            self.alpha_root * recon_loss_root
            + self.alpha_body * recon_loss_body
            + self.alpha_hand * recon_loss_hand
        )
        loss = recon_loss + commit_loss

        return recon, {
            "loss": loss,
            "recon_loss": recon_loss,
            "recon_root": recon_loss_root,
            "recon_body": recon_loss_body,
            "recon_hand": recon_loss_hand,
            "commit_loss": commit_loss,
            "commit_H": commit_H,
            "commit_B": commit_B,
        }, {"idxH": idxH, "idxB": idxB}


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