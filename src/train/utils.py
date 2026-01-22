import torch
from src.model.vqvae import H2VQ
import os
def build_model_from_args(args, device):
    if args.include_fingertips:
        body_in_dim = 263
        hand_in_dim = 480
    else:
        body_in_dim = 263
        hand_in_dim = 360
    model = H2VQ(
        T=args.T, 
        body_in_dim=body_in_dim,
        hand_in_dim=hand_in_dim,
        code_dim=args.code_dim,
        K=args.K,
        ema_decay=args.ema_decay,
        alpha_commit=args.alpha_commit,
        body_tokens_per_t=args.body_tokens_per_t,
        hand_tokens_per_t=args.hand_tokens_per_t,
        body_down=args.body_down,
        hand_down=args.hand_down,
        enc_type_B=args.enc_type_B,  # "xformer" or "cnn"
        enc_type_H=args.enc_type_H,  # "xformer" or "cnn"
        enc_use_attn_B=args.enc_use_attn_B,
        enc_use_attn_H=args.enc_use_attn_H,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        mlp_ratio=args.mlp_ratio,
        enc_use_pos=args.enc_use_pos,
        enc_post_mlp=args.enc_post_mlp,
        cnn_width_B=args.cnn_width_B,
        cnn_depth_B=args.cnn_depth_B,
        cnn_width_H=args.cnn_width_H,
        cnn_depth_H=args.cnn_depth_H,
        cnn_dilation_max=args.cnn_dilation_max,
        dec_hid=args.dec_hid,
        alpha_root=args.alpha_root,
        alpha_body=args.alpha_body,
        alpha_hand=args.alpha_hand,        
        use_root_loss=args.use_root_loss,
        include_fingertips=args.include_fingertips,
    ).to(device)
    return model

def _maybe_load_ckpt(path: str):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _safe_merge_args_from_ckpt(args, ckpt):
    if ckpt is None:
        return args

    ckpt_args = ckpt.get("args", None)
    if not isinstance(ckpt_args, dict):
        try:
            ckpt_args = vars(ckpt_args)
        except Exception:
            ckpt_args = None

    if not isinstance(ckpt_args, dict):
        return args

    keys_to_sync = [
        "K", "T", "include_fingertips",
        "model_type", "n_layers", "n_heads", "d_model", "d_ff",
        "mean_path", "std_path", "normalize",
        "joints_loss", "joints_loss_weight",
        "lr", "wd",
        "eval_every", "eval_num_save_samples", "eval_vis_dir", "eval_save_vis_every",
        "log_every", "ckpt_every",
        "project", "name",
        "epochs", "batch_size", "num_workers",
        "data_dir", "data_dir_eval",
    ]

    for k in keys_to_sync:
        if k in ckpt_args:
            try:
                args[k] = ckpt_args[k]
            except Exception:
                pass

    return args