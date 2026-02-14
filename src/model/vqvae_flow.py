# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ============================================================
# # EMA Quantizer
# # ============================================================
# class EMAQuantizer(nn.Module):
#     def __init__(self, K: int, D: int, ema_decay: float = 0.99, eps: float = 1e-5, reset_threshold: int = 1):
#         super().__init__()
#         self.K = K
#         self.D = D
#         self.ema_decay = ema_decay
#         self.eps = eps
#         self.reset_threshold = reset_threshold

#         embed = torch.randn(K, D) / math.sqrt(D)
#         self.register_buffer("codebook", embed)  # [K, D]
#         self.register_buffer("ema_cluster_size", torch.zeros(K))     # [K]
#         self.register_buffer("ema_codebook_sum", embed.clone())      # [K, D]

#     @torch.no_grad()
#     def _ema_update(self, onehot_assign: torch.Tensor, x_flat: torch.Tensor):
#         cluster_size = onehot_assign.sum(dim=0)          # [K]
#         code_sum = onehot_assign.t() @ x_flat            # [K, D]

#         self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
#         self.ema_codebook_sum.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)

#         n = self.ema_cluster_size.sum()
#         smoothed = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps) * n
#         new_codebook = self.ema_codebook_sum / smoothed.unsqueeze(1)
#         self.codebook.copy_(new_codebook)

#     @torch.no_grad()
#     def _code_reset(self, x_flat: torch.Tensor):
#         dead = self.ema_cluster_size < self.reset_threshold
#         num_dead = int(dead.sum().item())
#         if num_dead == 0:
#             return
#         idx = torch.randint(0, x_flat.size(0), (num_dead,), device=x_flat.device)
#         repl = x_flat[idx]
#         self.codebook[dead] = repl
#         self.ema_codebook_sum[dead] = repl
#         self.ema_cluster_size[dead] = self.reset_threshold

#     def forward(self, x: torch.Tensor, do_reset: bool = True):
#         # x: [..., D]
#         orig_shape = x.shape
#         assert orig_shape[-1] == self.D
#         x_flat = x.reshape(-1, self.D)

#         x2 = (x_flat ** 2).sum(dim=1, keepdim=True)              # [N,1]
#         e2 = (self.codebook ** 2).sum(dim=1).unsqueeze(0)        # [1,K]
#         xe = x_flat @ self.codebook.t()                          # [N,K]
#         dist = x2 + e2 - 2 * xe

#         indices = dist.argmin(dim=1)                             # [N]
#         x_q = self.codebook[indices].reshape(*orig_shape)
#         x_q_st = x + (x_q - x).detach()

#         if self.training:
#             with torch.no_grad():
#                 onehot = F.one_hot(indices, num_classes=self.K).type_as(x_flat)
#                 self._ema_update(onehot, x_flat)
#                 if do_reset:
#                     self._code_reset(x_flat)

#         return x_q_st, indices.reshape(orig_shape[:-1])

# # ============================================================
# # Helpers
# # ============================================================
# def build_1d_sincos_posemb(max_len: int, embed_dim: int, temperature: float = 10000.0):
#     pos = torch.arange(max_len, dtype=torch.float32)  # (T,)
#     assert embed_dim % 2 == 0
#     half = embed_dim // 2
#     omega = torch.arange(half, dtype=torch.float32) / half
#     omega = 1.0 / (temperature ** omega)
#     out = torch.einsum("t,d->td", pos, omega)
#     emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0)  # (1,T,D)
#     return emb

# def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
#     half = dim // 2
#     freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
#     args = t[:, None] * freqs[None, :] * 2 * math.pi
#     emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2 == 1:
#         emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
#     return emb

# def _group_norm(num_channels: int, max_groups: int = 8):
#     g = min(max_groups, num_channels)
#     while g > 1 and (num_channels % g) != 0:
#         g -= 1
#     return nn.GroupNorm(g, num_channels)

# # ============================================================
# # Encoder blocks
# # ============================================================
# class Mlp(nn.Module):
#     def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act=nn.GELU):
#         super().__init__()
#         hid = int(dim * mlp_ratio)
#         self.fc1 = nn.Linear(dim, hid)
#         self.fc2 = nn.Linear(hid, dim)
#         self.act = act()
#         self.drop = nn.Dropout(drop)
#     def forward(self, x):
#         x = self.drop(self.fc1(x))
#         x = self.act(x)
#         x = self.drop(self.fc2(x))
#         return x

# class Attn(nn.Module):
#     def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj_drop(self.proj(out))
#         return out

# class Block(nn.Module):
#     def __init__(self, dim, heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
#         super().__init__()
#         self.n1 = nn.LayerNorm(dim)
#         self.n2 = nn.LayerNorm(dim)
#         self.attn = Attn(dim, heads=heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
#     def forward(self, x):
#         x = x + self.attn(self.n1(x))
#         x = x + self.mlp(self.n2(x))
#         return x

# class ConvXFormerEncoder1D(nn.Module):
#     """
#     x: [B,T,Cin] -> [B,T',Cout], T' = T / r
#     """
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         num_frames: int,
#         temporal_compress: int,
#         use_attn: bool = True,
#         depth: int = 6,
#         heads: int = 8,
#         mlp_ratio: float = 4.0,
#         drop: float = 0.0,
#         attn_drop: float = 0.0,
#         use_pos: bool = True,
#         learnable_pos_emb: bool = False,
#         post_mlp: bool = True,
#     ):
#         super().__init__()
#         self.r = temporal_compress
#         self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=self.r, stride=self.r)

#         Tp = max(1, num_frames // self.r)
#         if use_pos:
#             pos = build_1d_sincos_posemb(Tp, embed_dim=out_dim)
#             self.pos = nn.Parameter(pos, requires_grad=learnable_pos_emb)
#         else:
#             self.pos = None

#         if (use_attn is False) or (depth <= 0):
#             self.blocks = nn.Identity()
#         else:
#             self.blocks = nn.Sequential(*[
#                 Block(out_dim, heads=heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
#                 for _ in range(depth)
#             ])

#         if post_mlp:
#             self.norm = nn.LayerNorm(out_dim)
#             self.post_mlp = Mlp(out_dim, mlp_ratio=mlp_ratio, drop=drop, act=nn.Tanh)
#         else:
#             self.norm = None
#             self.post_mlp = None

#     def forward(self, x: torch.Tensor):
#         x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
#         if self.pos is not None:
#             x = x + self.pos[:, :x.size(1)]
#         x = self.blocks(x)
#         if self.post_mlp is not None:
#             x = x + self.post_mlp(self.norm(x))
#         return x

# class ResConv1DBlock(nn.Module):
#     def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, drop: float = 0.0):
#         super().__init__()
#         pad = (kernel - 1) // 2 * dilation
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
#         self.gn1 = _group_norm(channels)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation)
#         self.gn2 = _group_norm(channels)
#         self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
#     def forward(self, x):
#         h = self.drop(F.gelu(self.gn1(self.conv1(x))))
#         h = self.drop(self.gn2(self.conv2(h)))
#         return F.gelu(x + h)

# class CNNEncoder1D(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         num_frames: int,
#         temporal_compress: int,
#         cnn_width: int = 512,
#         cnn_depth: int = 8,
#         cnn_kernel: int = 3,
#         dilation_cycle: bool = True,
#         dilation_max: int = 8,
#         drop: float = 0.0,
#     ):
#         super().__init__()
#         self.r = temporal_compress
#         self.stem = nn.Sequential(
#             nn.Conv1d(in_dim, cnn_width, kernel_size=self.r, stride=self.r),
#             _group_norm(cnn_width),
#             nn.GELU(),
#         )
#         blocks = []
#         for i in range(cnn_depth):
#             if dilation_cycle:
#                 d = 2 ** (i % int(math.log2(dilation_max) + 1))
#                 d = min(d, dilation_max)
#             else:
#                 d = 1
#             blocks.append(ResConv1DBlock(cnn_width, kernel=cnn_kernel, dilation=d, drop=drop))
#         self.blocks = nn.Sequential(*blocks)
#         self.proj = nn.Conv1d(cnn_width, out_dim, kernel_size=1)

#     def forward(self, x: torch.Tensor):
#         x = x.permute(0, 2, 1)
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.proj(x)
#         x = x.permute(0, 2, 1).contiguous()
#         return x

# # ============================================================
# # Flow decoder (DiT-ish, cross-attn cond)
# # ============================================================
# class CrossAttn(nn.Module):
#     def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, cond):
#         B, N, C = x.shape
#         _, M, _ = cond.shape
#         q = self.q(x).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
#         kv = self.kv(cond).reshape(B, M, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj_drop(self.proj(out))
#         return out

# class AdaLN(nn.Module):
#     def __init__(self, dim, t_dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim, elementwise_affine=False)
#         self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, dim * 2))
#     def forward(self, x, t_emb):
#         h = self.norm(x)
#         ss = self.to_scale_shift(t_emb)
#         scale, shift = ss.chunk(2, dim=-1)
#         return h * (1 + scale[:, None, :]) + shift[:, None, :]

# class DiTBlock(nn.Module):
#     def __init__(self, dim, heads, t_dim, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
#         super().__init__()
#         self.self_attn = Attn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
#         self.cross_attn = CrossAttn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
#         self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
#         self.adaln1 = AdaLN(dim, t_dim)
#         self.adaln2 = AdaLN(dim, t_dim)
#         self.adaln3 = AdaLN(dim, t_dim)
#     def forward(self, x, t_emb, cond):
#         x = x + self.self_attn(self.adaln1(x, t_emb))
#         x = x + self.cross_attn(self.adaln2(x, t_emb), cond)
#         x = x + self.mlp(self.adaln3(x, t_emb))
#         return x

# class FlowDecoder1D(nn.Module):
#     def __init__(
#         self,
#         x_dim: int,
#         cond_dim: int,
#         model_dim: int = 512,
#         depth: int = 8,
#         heads: int = 8,
#         mlp_ratio: float = 4.0,
#         drop: float = 0.0,
#         attn_drop: float = 0.0,
#         t_dim: int = 512,
#         use_x_pos: bool = True,
#         max_T: int = 2048,
#     ):
#         super().__init__()
#         self.x_in = nn.Linear(x_dim, model_dim)
#         self.cond_in = nn.Linear(cond_dim, model_dim)

#         if use_x_pos:
#             pos = build_1d_sincos_posemb(max_T, embed_dim=model_dim)
#             self.register_buffer("x_pos", pos, persistent=False)
#         else:
#             self.x_pos = None

#         self.t_mlp = nn.Sequential(
#             nn.Linear(t_dim, t_dim),
#             nn.SiLU(),
#             nn.Linear(t_dim, t_dim),
#         )
#         self._t_dim = t_dim

#         self.blocks = nn.ModuleList([
#             DiTBlock(model_dim, heads=heads, t_dim=t_dim, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
#             for _ in range(depth)
#         ])

#         self.out_norm = nn.LayerNorm(model_dim)
#         self.x_out = nn.Linear(model_dim, x_dim)

#     def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor):
#         B, T, _ = x_t.shape
#         x = self.x_in(x_t)
#         if self.x_pos is not None:
#             x = x + self.x_pos[:, :T]
#         cond = self.cond_in(cond_tokens)

#         t_emb = timestep_embedding(t, dim=self._t_dim)
#         t_emb = self.t_mlp(t_emb)

#         for blk in self.blocks:
#             x = blk(x, t_emb, cond)

#         v = self.x_out(self.out_norm(x))
#         return v

# # ============================================================
# # Flow-only H2VQ
# # ============================================================
# class H2VQFlow(nn.Module):
#     def __init__(
#         self,
#         T: int,
#         body_in_dim: int,
#         hand_in_dim: int,

#         code_dim: int = 512,
#         K: int = 512,
#         ema_decay: float = 0.99,
#         alpha_commit: float = 0.02,
#         lambda_flow: float = 1.0,
#         lambda_entropy: float = 1e-3,

#         body_tokens_per_t: int = 2,
#         hand_tokens_per_t: int = 4,
#         body_down: int = 4,
#         hand_down: int = 4,

#         enc_type_B: str = "xformer",
#         enc_type_H: str = "xformer",

#         # xformer enc
#         enc_depth: int = 6,
#         enc_heads: int = 8,
#         enc_mlp_ratio: float = 4.0,
#         enc_drop: float = 0.0,
#         enc_attn_drop: float = 0.0,
#         enc_use_pos: bool = True,
#         enc_post_mlp: bool = True,
#         enc_use_attn_B: bool = True,
#         enc_use_attn_H: bool = True,

#         # cnn enc
#         cnn_width_B: int = 512,
#         cnn_depth_B: int = 8,
#         cnn_width_H: int = 512,
#         cnn_depth_H: int = 8,
#         cnn_kernel: int = 3,
#         cnn_dilation_max: int = 8,
#         cnn_drop: float = 0.0,

#         # fuse
#         use_fuse: bool = True,

#         # flow decoder
#         flow_model_dim: int = 512,
#         flow_depth: int = 8,
#         flow_heads: int = 8,
#         flow_mlp_ratio: float = 4.0,
#         flow_drop: float = 0.0,
#         flow_attn_drop: float = 0.0,
#         flow_t_dim: int = 512,
#         flow_use_x_pos: bool = True,

#         # part weights + root selection
#         alpha_root: float = 1.0,
#         alpha_body: float = 1.0,
#         alpha_hand: float = 1.0,
#         use_root_loss: bool = True,     # True: [:4], False: only root_keep_idx
#         root_keep_idx=(0, 3),

#         # IMPORTANT: also mask decoder input dims where weight==0
#         mask_input_dims: bool = True,

#         # quant reset
#         do_reset: bool = True,
#     ):
#         super().__init__()
#         assert enc_type_B in ["xformer", "cnn"]
#         assert enc_type_H in ["xformer", "cnn"]

#         self.T = T
#         self.body_in_dim = body_in_dim
#         self.hand_in_dim = hand_in_dim
#         self.x_dim = body_in_dim + hand_in_dim

#         self.code_dim = code_dim
#         self.K = K
#         self.body_tokens_per_t = body_tokens_per_t
#         self.hand_tokens_per_t = hand_tokens_per_t
#         self.body_down = body_down
#         self.hand_down = hand_down
#         self.use_fuse = use_fuse

#         self.alpha_commit = alpha_commit
#         self.lambda_flow = lambda_flow
#         self.lambda_entropy = lambda_entropy

#         self.alpha_root = alpha_root
#         self.alpha_body = alpha_body
#         self.alpha_hand = alpha_hand
#         self.use_root_loss = use_root_loss
#         self.root_keep_idx = tuple(root_keep_idx)
#         self.mask_input_dims = mask_input_dims
#         self.do_reset = do_reset

#         hand_out = hand_tokens_per_t * code_dim
#         body_out = body_tokens_per_t * code_dim

#         # encoders
#         if enc_type_H == "xformer":
#             self.encH = ConvXFormerEncoder1D(
#                 in_dim=hand_in_dim, out_dim=hand_out, num_frames=T, temporal_compress=hand_down,
#                 use_attn=enc_use_attn_H, depth=enc_depth, heads=enc_heads,
#                 mlp_ratio=enc_mlp_ratio, drop=enc_drop, attn_drop=enc_attn_drop,
#                 use_pos=enc_use_pos, post_mlp=enc_post_mlp,
#             )
#         else:
#             self.encH = CNNEncoder1D(
#                 in_dim=hand_in_dim, out_dim=hand_out, num_frames=T, temporal_compress=hand_down,
#                 cnn_width=cnn_width_H, cnn_depth=cnn_depth_H, cnn_kernel=cnn_kernel,
#                 dilation_max=cnn_dilation_max, drop=cnn_drop,
#             )

#         if enc_type_B == "xformer":
#             self.encB = ConvXFormerEncoder1D(
#                 in_dim=body_in_dim, out_dim=body_out, num_frames=T, temporal_compress=body_down,
#                 use_attn=enc_use_attn_B, depth=enc_depth, heads=enc_heads,
#                 mlp_ratio=enc_mlp_ratio, drop=enc_drop, attn_drop=enc_attn_drop,
#                 use_pos=enc_use_pos, post_mlp=enc_post_mlp,
#             )
#         else:
#             self.encB = CNNEncoder1D(
#                 in_dim=body_in_dim, out_dim=body_out, num_frames=T, temporal_compress=body_down,
#                 cnn_width=cnn_width_B, cnn_depth=cnn_depth_B, cnn_kernel=cnn_kernel,
#                 dilation_max=cnn_dilation_max, drop=cnn_drop,
#             )

#         # quantizers
#         self.qH = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
#         self.qB = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)

#         # fuse
#         self.hand_proj = nn.Linear(hand_out, hand_out)
#         self.fuse_proj = nn.Linear(hand_out + body_out, body_out)

#         # flow decoder
#         self.flow = FlowDecoder1D(
#             x_dim=self.x_dim,
#             cond_dim=self.code_dim,
#             model_dim=flow_model_dim,
#             depth=flow_depth,
#             heads=flow_heads,
#             mlp_ratio=flow_mlp_ratio,
#             drop=flow_drop,
#             attn_drop=flow_attn_drop,
#             t_dim=flow_t_dim,
#             use_x_pos=flow_use_x_pos,
#             max_T=T,
#         )

#         # dim weighting mask
#         w = torch.ones(self.x_dim)

#         if use_root_loss:
#             w[:4] = alpha_root
#         else:
#             w[:4] = 0.0
#             for k in self.root_keep_idx:
#                 w[k] = alpha_root

#         w[4:263] = alpha_body
#         w[263:]  = alpha_hand

#         self.register_buffer("dim_weight", w, persistent=False)
#         self.register_buffer("dim_keep", (w > 0).float(), persistent=False)

#     def _split_tokens(self, z_seq: torch.Tensor, tokens_per_t: int):
#         B, Tp, C = z_seq.shape
#         D = self.code_dim
#         return z_seq.view(B, Tp, tokens_per_t, D)

#     def _merge_tokens(self, z_tok: torch.Tensor):
#         B, Tp, tok, D = z_tok.shape
#         return z_tok.view(B, Tp, tok * D)

#     def _cond_from_ids(self, idxH: torch.Tensor, idxB: torch.Tensor):
#         cH = self.qH.codebook[idxH]  # [B,T',Ht,D]
#         cB = self.qB.codebook[idxB]  # [B,T',Bt,D]
#         B, Tp, Ht, D = cH.shape
#         _, _, Bt, _ = cB.shape
#         return torch.cat([cB.reshape(B, Tp * Bt, D), cH.reshape(B, Tp * Ht, D)], dim=1)

#     def _entropy_term(self, idx: torch.Tensor):
#         x = idx.reshape(-1)
#         onehot = F.one_hot(x, num_classes=self.K).float()
#         p = onehot.mean(dim=0)
#         return (p * torch.log(p + 1e-9)).sum()

#     def forward(self, mB: torch.Tensor, mH: torch.Tensor):
#         x0 = torch.cat([mB, mH], dim=-1)  # [B,T,623]
#         B, T, _ = x0.shape

#         # encode
#         zH = self.encH(mH)
#         zB = self.encB(mB)
#         Tm = min(zH.size(1), zB.size(1))
#         zH, zB = zH[:, :Tm], zB[:, :Tm]

#         # quantize hand
#         zH_tok = self._split_tokens(zH, self.hand_tokens_per_t)
#         zH_q_tok, idxH = self.qH(zH_tok, do_reset=self.do_reset)
#         zH_q = self._merge_tokens(zH_q_tok)

#         # fuse -> quantize body
#         if self.use_fuse:
#             zH_proj = self.hand_proj(zH_q)
#             z_fused = self.fuse_proj(torch.cat([zH_proj, zB], dim=-1))
#         else:
#             z_fused = zB

#         zB_tok = self._split_tokens(z_fused, self.body_tokens_per_t)
#         zB_q_tok, idxB = self.qB(zB_tok, do_reset=self.do_reset)

#         # commit
#         commit_H = F.mse_loss(zH_tok, zH_q_tok.detach())
#         commit_B = F.mse_loss(zB_tok, zB_q_tok.detach())
#         commit_loss = self.alpha_commit * (commit_H + commit_B)

#         # entropy
#         entH = self._entropy_term(idxH)
#         entB = self._entropy_term(idxB)
#         entropy_loss = self.lambda_entropy * (entH + entB)

#         # flow matching
#         cond = self._cond_from_ids(idxH, idxB)
#         t = torch.rand(B, device=x0.device)
#         x1 = torch.randn_like(x0)
#         x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
#         v_star = x1 - x0

#         if self.mask_input_dims:
#             x_t = x_t * self.dim_keep[None, None, :]

#         v_pred = self.flow(x_t, t, cond)

#         w = self.dim_weight[None, None, :]
#         flow_loss = self.lambda_flow * torch.mean(((v_pred - v_star) * w) ** 2)

#         loss = flow_loss + commit_loss + entropy_loss

#         return None, {
#             "loss": loss,
#             "flow_loss": flow_loss,
#             "commit_loss": commit_loss,
#             "entropy_loss": entropy_loss,
#             "commit_H": commit_H,
#             "commit_B": commit_B,
#             "entH_raw": entH,
#             "entB_raw": entB,
#         }, {"idxH": idxH, "idxB": idxB}

#     @torch.no_grad()
#     def sample_from_ids(self, idxH: torch.Tensor, idxB: torch.Tensor, target_T: int, steps: int = 30, solver: str = "heun"):
#         cond = self._cond_from_ids(idxH, idxB)
#         B = idxH.size(0)
#         x = torch.randn(B, target_T, self.x_dim, device=idxH.device)

#         ts = torch.linspace(1.0, 0.0, steps + 1, device=idxH.device)
#         for i in range(steps):
#             t0 = ts[i].expand(B)
#             t1 = ts[i + 1].expand(B)
#             dt = (t1 - t0)

#             x_in = x
#             if self.mask_input_dims:
#                 x_in = x_in * self.dim_keep[None, None, :]

#             v0 = self.flow(x_in, t0, cond)

#             if solver == "euler":
#                 x = x + dt[:, None, None] * v0
#             elif solver == "heun":
#                 x_e = x + dt[:, None, None] * v0
#                 x_e_in = x_e
#                 if self.mask_input_dims:
#                     x_e_in = x_e_in * self.dim_keep[None, None, :]
#                 v1 = self.flow(x_e_in, t1, cond)
#                 x = x + dt[:, None, None] * 0.5 * (v0 + v1)
#             else:
#                 raise ValueError(f"unknown solver: {solver}")

#         return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# EMA Quantizer
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
def build_1d_sincos_posemb(max_len: int, embed_dim: int, temperature: float = 10000.0):
    pos = torch.arange(max_len, dtype=torch.float32)  # (T,)
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / half
    omega = 1.0 / (temperature ** omega)
    out = torch.einsum("t,d->td", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0)  # (1,T,D)
    return emb

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
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)             # [T, head_dim]
        # shape: [1, 1, T, head_dim] for broadcasting with [B, heads, T, head_dim]
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
# Encoder blocks
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
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
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=self.r, stride=self.r)

        Tp = max(1, num_frames // self.r)
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
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.pos is not None:
            x = x + self.pos[:, :x.size(1)]
        x = self.blocks(x)
        if self.post_mlp is not None:
            x = x + self.post_mlp(self.norm(x))
        return x

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
        h = self.drop(F.gelu(self.gn1(self.conv1(x))))
        h = self.drop(self.gn2(self.conv2(h)))
        return F.gelu(x + h)

class CNNEncoder1D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_frames: int,
        temporal_compress: int,
        cnn_width: int = 512,
        cnn_depth: int = 8,
        cnn_kernel: int = 3,
        dilation_cycle: bool = True,
        dilation_max: int = 8,
        drop: float = 0.0,
    ):
        super().__init__()
        self.r = temporal_compress
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, cnn_width, kernel_size=self.r, stride=self.r),
            _group_norm(cnn_width),
            nn.GELU(),
        )
        blocks = []
        for i in range(cnn_depth):
            if dilation_cycle:
                d = 2 ** (i % int(math.log2(dilation_max) + 1))
                d = min(d, dilation_max)
            else:
                d = 1
            blocks.append(ResConv1DBlock(cnn_width, kernel=cnn_kernel, dilation=d, drop=drop))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(cnn_width, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

# ============================================================
# Flow decoder (DiT-ish, cross-attn cond)
# ============================================================
class CrossAttn(nn.Module):
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
        self.cross_attn = CrossAttn(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
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

        # Position encoding: RoPE or absolute sinusoidal (mutually exclusive)
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
            DiTBlock(model_dim, heads=heads, t_dim=t_dim, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(model_dim)
        self.x_out = nn.Linear(model_dim, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor):
        B, T, _ = x_t.shape
        x = self.x_in(x_t)
        if self.x_pos is not None:
            x = x + self.x_pos[:, :T]

        # Compute RoPE cos/sin for this sequence length
        rope = self.rope(T) if self.rope is not None else None

        cond = self.cond_in(cond_tokens)

        t_emb = timestep_embedding(t, dim=self._t_dim)
        t_emb = self.t_mlp(t_emb)

        for blk in self.blocks:
            x = blk(x, t_emb, cond, rope=rope)

        v = self.x_out(self.out_norm(x))
        return v

# ============================================================
# Flexible H2VQFlow with multiple conditioning strategies
# ============================================================
class H2VQFlow(nn.Module):
    """
    cond_type options:
      "baseline"      : 元のまま (body_tokens + hand_tokens concat)
      "part_embed"    : Part embedding を追加 (0=body, 1=hand)
      "marker"        : Part marker (learnable vectors)
      "decoder_separate": Body/Hand で decoder を分離
    """
    def __init__(
        self,
        T: int,
        body_in_dim: int,
        hand_in_dim: int,

        code_dim: int = 512,
        K: int = 512,
        ema_decay: float = 0.99,
        alpha_commit: float = 0.02,
        lambda_flow: float = 1.0,
        lambda_entropy: float = 1e-3,

        body_tokens_per_t: int = 2,
        hand_tokens_per_t: int = 4,
        body_down: int = 4,
        hand_down: int = 4,

        enc_type_B: str = "xformer",
        enc_type_H: str = "xformer",

        # xformer enc
        enc_depth: int = 6,
        enc_heads: int = 8,
        enc_mlp_ratio: float = 4.0,
        enc_drop: float = 0.0,
        enc_attn_drop: float = 0.0,
        enc_use_pos: bool = True,
        enc_post_mlp: bool = True,
        enc_use_attn_B: bool = True,
        enc_use_attn_H: bool = True,

        # cnn enc
        cnn_width_B: int = 512,
        cnn_depth_B: int = 8,
        cnn_width_H: int = 512,
        cnn_depth_H: int = 8,
        cnn_kernel: int = 3,
        cnn_dilation_max: int = 8,
        cnn_drop: float = 0.0,

        # fuse
        use_fuse: bool = True,

        # flow decoder
        flow_model_dim: int = 512,
        flow_depth: int = 8,
        flow_heads: int = 8,
        flow_mlp_ratio: float = 4.0,
        flow_drop: float = 0.0,
        flow_attn_drop: float = 0.0,
        flow_t_dim: int = 512,
        flow_use_x_pos: bool = True,
        flow_use_rope: bool = False,

        # part weights + root selection
        alpha_root: float = 1.0,
        alpha_body: float = 1.0,
        alpha_hand: float = 1.0,
        use_root_loss: bool = True,
        root_keep_idx=(0, 3),

        # IMPORTANT: also mask decoder input dims where weight==0
        mask_input_dims: bool = True,

        # quant reset
        do_reset: bool = True,

        # ========== NEW: Conditioning strategy ==========
        cond_type: str = "baseline",  # "baseline", "part_embed", "marker", "decoder_separate"
    ):
        super().__init__()
        assert enc_type_B in ["xformer", "cnn"]
        assert enc_type_H in ["xformer", "cnn"]
        assert cond_type in ["baseline", "part_embed", "marker", "decoder_separate"]

        self.T = T
        self.body_in_dim = body_in_dim
        self.hand_in_dim = hand_in_dim
        self.x_dim = body_in_dim + hand_in_dim

        self.code_dim = code_dim
        self.K = K
        self.body_tokens_per_t = body_tokens_per_t
        self.hand_tokens_per_t = hand_tokens_per_t
        self.body_down = body_down
        self.hand_down = hand_down
        self.use_fuse = use_fuse

        self.alpha_commit = alpha_commit
        self.lambda_flow = lambda_flow
        self.lambda_entropy = lambda_entropy

        self.alpha_root = alpha_root
        self.alpha_body = alpha_body
        self.alpha_hand = alpha_hand
        self.use_root_loss = use_root_loss
        self.root_keep_idx = tuple(root_keep_idx)
        self.mask_input_dims = mask_input_dims
        self.do_reset = do_reset
        self.cond_type = cond_type

        hand_out = hand_tokens_per_t * code_dim
        body_out = body_tokens_per_t * code_dim

        # ========== Encoders ==========
        if enc_type_H == "xformer":
            self.encH = ConvXFormerEncoder1D(
                in_dim=hand_in_dim, out_dim=hand_out, num_frames=T, temporal_compress=hand_down,
                use_attn=enc_use_attn_H, depth=enc_depth, heads=enc_heads,
                mlp_ratio=enc_mlp_ratio, drop=enc_drop, attn_drop=enc_attn_drop,
                use_pos=enc_use_pos, post_mlp=enc_post_mlp,
            )
        else:
            self.encH = CNNEncoder1D(
                in_dim=hand_in_dim, out_dim=hand_out, num_frames=T, temporal_compress=hand_down,
                cnn_width=cnn_width_H, cnn_depth=cnn_depth_H, cnn_kernel=cnn_kernel,
                dilation_max=cnn_dilation_max, drop=cnn_drop,
            )

        if enc_type_B == "xformer":
            self.encB = ConvXFormerEncoder1D(
                in_dim=body_in_dim, out_dim=body_out, num_frames=T, temporal_compress=body_down,
                use_attn=enc_use_attn_B, depth=enc_depth, heads=enc_heads,
                mlp_ratio=enc_mlp_ratio, drop=enc_drop, attn_drop=enc_attn_drop,
                use_pos=enc_use_pos, post_mlp=enc_post_mlp,
            )
        else:
            self.encB = CNNEncoder1D(
                in_dim=body_in_dim, out_dim=body_out, num_frames=T, temporal_compress=body_down,
                cnn_width=cnn_width_B, cnn_depth=cnn_depth_B, cnn_kernel=cnn_kernel,
                dilation_max=cnn_dilation_max, drop=cnn_drop,
            )

        # ========== Quantizers ==========
        self.qH = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)
        self.qB = EMAQuantizer(K=K, D=code_dim, ema_decay=ema_decay)

        # ========== Fuse ==========
        self.hand_proj = nn.Linear(hand_out, hand_out)
        self.fuse_proj = nn.Linear(hand_out + body_out, body_out)

        # ========== Conditioning Strategy Setup ==========
        if cond_type == "part_embed":
            # Part Embedding (ID-based)
            self.part_embed = nn.Embedding(2, code_dim)  # 0=body, 1=hand

        elif cond_type == "marker":
            # Learnable Part Markers
            self.body_marker = nn.Parameter(torch.randn(1, 1, code_dim))
            self.hand_marker = nn.Parameter(torch.randn(1, 1, code_dim))

        elif cond_type == "decoder_separate":
            # Separate decoders for body and hand, cross-conditioned
            self.body_cond_marker = nn.Parameter(torch.randn(1, 1, code_dim) * 0.02)
            self.hand_cond_marker = nn.Parameter(torch.randn(1, 1, code_dim) * 0.02)
            # Temporal pos emb for cond tokens: [1, max_Tp, 1, D] — broadcast over tokens_per_t
            # Allocate for max possible Tp, slice at runtime to handle T-1 etc.
            max_Tp = T
            cond_temporal_pos = build_1d_sincos_posemb(max_Tp, embed_dim=code_dim)  # [1, max_Tp, D]
            self.register_buffer("cond_temporal_pos", cond_temporal_pos.unsqueeze(2), persistent=False)  # [1, max_Tp, 1, D]
            cond_dim_B = code_dim
            cond_dim_H = code_dim

            self.flow_body = FlowDecoder1D(
                x_dim=body_in_dim,
                cond_dim=cond_dim_B,
                model_dim=flow_model_dim,
                depth=flow_depth,
                heads=flow_heads,
                mlp_ratio=flow_mlp_ratio,
                drop=flow_drop,
                attn_drop=flow_attn_drop,
                t_dim=flow_t_dim,
                use_x_pos=flow_use_x_pos,
                max_T=T,
                use_rope=flow_use_rope,
            )
            self.flow_hand = FlowDecoder1D(
                x_dim=hand_in_dim,
                cond_dim=cond_dim_H,
                model_dim=flow_model_dim,
                depth=flow_depth,
                heads=flow_heads,
                mlp_ratio=flow_mlp_ratio,
                drop=flow_drop,
                attn_drop=flow_attn_drop,
                t_dim=flow_t_dim,
                use_x_pos=flow_use_x_pos,
                max_T=T,
                use_rope=flow_use_rope,
            )

        else:
            # cond_type == "baseline"
            # Single flow decoder (original behavior)
            self.flow = FlowDecoder1D(
                x_dim=self.x_dim,
                cond_dim=self.code_dim,
                model_dim=flow_model_dim,
                depth=flow_depth,
                heads=flow_heads,
                mlp_ratio=flow_mlp_ratio,
                drop=flow_drop,
                attn_drop=flow_attn_drop,
                t_dim=flow_t_dim,
                use_x_pos=flow_use_x_pos,
                max_T=T,
                use_rope=flow_use_rope,
            )

        # ========== Dim weighting mask ==========
        w = torch.ones(self.x_dim)

        if use_root_loss:
            w[:4] = alpha_root
        else:
            w[:4] = 0.0
            for k in self.root_keep_idx:
                w[k] = alpha_root

        w[4:263] = alpha_body
        w[263:]  = alpha_hand

        self.register_buffer("dim_weight", w, persistent=False)
        self.register_buffer("dim_keep", (w > 0).float(), persistent=False)

    def _split_tokens(self, z_seq: torch.Tensor, tokens_per_t: int):
        B, Tp, C = z_seq.shape
        D = self.code_dim
        return z_seq.view(B, Tp, tokens_per_t, D)

    def _merge_tokens(self, z_tok: torch.Tensor):
        B, Tp, tok, D = z_tok.shape
        return z_tok.view(B, Tp, tok * D)

    def _cond_from_ids(self, idxH: torch.Tensor, idxB: torch.Tensor):
        """Create conditioning tokens based on cond_type"""
        cH = self.qH.codebook[idxH]  # [B,T',Ht,D]
        cB = self.qB.codebook[idxB]  # [B,T',Bt,D]
        B, Tp, Ht, D = cH.shape
        _, _, Bt, _ = cB.shape

        cB_flat = cB.reshape(B, Tp * Bt, D)
        cH_flat = cH.reshape(B, Tp * Ht, D)

        if self.cond_type == "part_embed":
            # Add part embedding (0=body, 1=hand)
            cB_flat = cB_flat + self.part_embed(torch.tensor(0, device=cB.device))
            cH_flat = cH_flat + self.part_embed(torch.tensor(1, device=cH.device))

        elif self.cond_type == "marker":
            # Add learnable markers
            cB_flat = cB_flat + self.body_marker
            cH_flat = cH_flat + self.hand_marker

        # For all types, return concatenated tokens
        if self.cond_type == "decoder_separate":
            # Add temporal pos emb before flattening
            # cB: [B, Tp, Bt, D], cH: [B, Tp, Ht, D]
            # cond_temporal_pos: [1, max_Tp, 1, D] — slice to actual Tp
            cB = cB + self.cond_temporal_pos[:, :Tp]
            cH = cH + self.cond_temporal_pos[:, :Tp]
            cB_flat = cB.reshape(B, Tp * Bt, D)
            cH_flat = cH.reshape(B, Tp * Ht, D)
            # Add part markers and concatenate for cross-conditioning
            cB_flat = cB_flat + self.body_cond_marker
            cH_flat = cH_flat + self.hand_cond_marker
            return torch.cat([cB_flat, cH_flat], dim=1)  # [B, Tp*(Bt+Ht), D]
        else:
            # Concatenate for single decoder
            return torch.cat([cB_flat, cH_flat], dim=1)

    def _entropy_term(self, idx: torch.Tensor):
        x = idx.reshape(-1)
        onehot = F.one_hot(x, num_classes=self.K).float()
        p = onehot.mean(dim=0)
        return (p * torch.log(p + 1e-9)).sum()

    def forward(self, mB: torch.Tensor, mH: torch.Tensor):
        x0 = torch.cat([mB, mH], dim=-1)  # [B,T,623]
        B, T, _ = x0.shape

        # encode
        zH = self.encH(mH)
        zB = self.encB(mB)
        Tm = min(zH.size(1), zB.size(1))
        zH, zB = zH[:, :Tm], zB[:, :Tm]

        # quantize hand
        zH_tok = self._split_tokens(zH, self.hand_tokens_per_t)
        zH_q_tok, idxH = self.qH(zH_tok, do_reset=self.do_reset)
        zH_q = self._merge_tokens(zH_q_tok)

        # fuse -> quantize body
        if self.use_fuse:
            zH_proj = self.hand_proj(zH_q)
            z_fused = self.fuse_proj(torch.cat([zH_proj, zB], dim=-1))
        else:
            z_fused = zB

        zB_tok = self._split_tokens(z_fused, self.body_tokens_per_t)
        zB_q_tok, idxB = self.qB(zB_tok, do_reset=self.do_reset)

        # commit
        commit_H = F.mse_loss(zH_tok, zH_q_tok.detach())
        commit_B = F.mse_loss(zB_tok, zB_q_tok.detach())
        commit_loss = self.alpha_commit * (commit_H + commit_B)

        # entropy
        entH = self._entropy_term(idxH)
        entB = self._entropy_term(idxB)
        entropy_loss = self.lambda_entropy * (entH + entB)

        # ========== Flow matching ==========
        t = torch.rand(B, device=x0.device)
        x1 = torch.randn_like(x0)
        x_t = (1.0 - t)[:, None, None] * x0 + t[:, None, None] * x1
        v_star = x1 - x0

        if self.mask_input_dims:
            x_t = x_t * self.dim_keep[None, None, :]

        if self.cond_type == "decoder_separate":
            # Separate flows for body and hand, cross-conditioned
            cond_all = self._cond_from_ids(idxH, idxB)

            v_pred_B = self.flow_body(x_t[:, :, :self.body_in_dim], t, cond_all)
            v_pred_H = self.flow_hand(x_t[:, :, self.body_in_dim:], t, cond_all)
            v_pred = torch.cat([v_pred_B, v_pred_H], dim=-1)

            w_B = self.dim_weight[None, None, :self.body_in_dim]
            w_H = self.dim_weight[None, None, self.body_in_dim:]

            flow_loss_B = torch.mean(
                ((v_pred_B - (v_star[:, :, :self.body_in_dim])) * w_B) ** 2
            )
            flow_loss_H = torch.mean(
                ((v_pred_H - (v_star[:, :, self.body_in_dim:])) * w_H) ** 2
            )
            flow_loss = self.lambda_flow * (flow_loss_B + flow_loss_H)

        else:
            # Single flow decoder
            cond = self._cond_from_ids(idxH, idxB)
            v_pred = self.flow(x_t, t, cond)

            w = self.dim_weight[None, None, :]
            flow_loss = self.lambda_flow * torch.mean(((v_pred - v_star) * w) ** 2)

        loss = flow_loss + commit_loss + entropy_loss

        return None, {
            "loss": loss,
            "flow_loss": flow_loss,
            "commit_loss": commit_loss,
            "entropy_loss": entropy_loss,
            "commit_H": commit_H,
            "commit_B": commit_B,
            "entH_raw": entH,
            "entB_raw": entB,
        }, {"idxH": idxH, "idxB": idxB}

    @torch.no_grad()
    def sample_from_ids(self, idxH: torch.Tensor, idxB: torch.Tensor, target_T: int, steps: int = 30, solver: str = "heun"):
        B = idxH.size(0)

        if self.cond_type == "decoder_separate":
            # Separate sampling for body and hand, cross-conditioned
            cond_all = self._cond_from_ids(idxH, idxB)

            x_B = torch.randn(B, target_T, self.body_in_dim, device=idxH.device)
            x_H = torch.randn(B, target_T, self.hand_in_dim, device=idxH.device)

            ts = torch.linspace(1.0, 0.0, steps + 1, device=idxH.device)
            for i in range(steps):
                t0 = ts[i].expand(B)
                t1 = ts[i + 1].expand(B)
                dt = (t1 - t0)

                # Body
                x_B_in = x_B
                if self.mask_input_dims:
                    x_B_in = x_B_in * self.dim_keep[None, None, :self.body_in_dim]
                v0_B = self.flow_body(x_B_in, t0, cond_all)

                # Hand
                x_H_in = x_H
                if self.mask_input_dims:
                    x_H_in = x_H_in * self.dim_keep[None, None, self.body_in_dim:]
                v0_H = self.flow_hand(x_H_in, t0, cond_all)

                if solver == "euler":
                    x_B = x_B + dt[:, None, None] * v0_B
                    x_H = x_H + dt[:, None, None] * v0_H

                elif solver == "heun":
                    x_B_e = x_B + dt[:, None, None] * v0_B
                    x_B_e_in = x_B_e
                    if self.mask_input_dims:
                        x_B_e_in = x_B_e_in * self.dim_keep[None, None, :self.body_in_dim]
                    v1_B = self.flow_body(x_B_e_in, t1, cond_all)
                    x_B = x_B + dt[:, None, None] * 0.5 * (v0_B + v1_B)

                    x_H_e = x_H + dt[:, None, None] * v0_H
                    x_H_e_in = x_H_e
                    if self.mask_input_dims:
                        x_H_e_in = x_H_e_in * self.dim_keep[None, None, self.body_in_dim:]
                    v1_H = self.flow_hand(x_H_e_in, t1, cond_all)
                    x_H = x_H + dt[:, None, None] * 0.5 * (v0_H + v1_H)

                else:
                    raise ValueError(f"unknown solver: {solver}")

            return torch.cat([x_B, x_H], dim=-1)

        else:
            # Single decoder sampling (original behavior)
            cond = self._cond_from_ids(idxH, idxB)
            x = torch.randn(B, target_T, self.x_dim, device=idxH.device)

            ts = torch.linspace(1.0, 0.0, steps + 1, device=idxH.device)
            for i in range(steps):
                t0 = ts[i].expand(B)
                t1 = ts[i + 1].expand(B)
                dt = (t1 - t0)

                x_in = x
                if self.mask_input_dims:
                    x_in = x_in * self.dim_keep[None, None, :]

                v0 = self.flow(x_in, t0, cond)

                if solver == "euler":
                    x = x + dt[:, None, None] * v0
                elif solver == "heun":
                    x_e = x + dt[:, None, None] * v0
                    x_e_in = x_e
                    if self.mask_input_dims:
                        x_e_in = x_e_in * self.dim_keep[None, None, :]
                    v1 = self.flow(x_e_in, t1, cond)
                    x = x + dt[:, None, None] * 0.5 * (v0 + v1)
                else:
                    raise ValueError(f"unknown solver: {solver}")

            return x