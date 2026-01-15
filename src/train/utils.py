import torch
from src.model.vqvae import H2VQ
def build_model_from_args(args, device):
    model = H2VQ(
        T=args.T, 
        body_in_dim=args.body_in_dim,
        hand_in_dim=args.hand_in_dim,
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