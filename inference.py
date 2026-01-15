from src.train.utils import build_model_from_args
from src.dataset.infer_loader import MotionInferenceDataset
from src.dataset.collate import collate_stack
from torch.utils.data import DataLoader
import torch
import argparse
import os
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="/large/naru/EgoHand/BodyTokenize/runs/cnn_large_token_8_1000/config.yaml")
    ap.add_argument("--ckpt", type=str, default="/large/naru/EgoHand/BodyTokenize/runs/cnn_large_token_8_1000/ckpt_epoch980.pt")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--video_base_dir", type=str, default="/large/naru/EgoHand/data/takes_clipped/videos")
    ap.add_argument("--data_save_dir", type=str, default="/large/naru/EgoHand/data/takes_clipped")
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(args, device)
    ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    video_paths = glob.glob(os.path.join(args_cli.video_base_dir, "**/*.mp4"), recursive=True)
    video_paths.sort()

    for video_path in tqdm(video_paths, desc="Processing videos"):
        sample_name = video_path.split("/")[-2]
        start = video_path.split("/")[-1].split(".")[0].split("___")[0]
        end = video_path.split("/")[-1].split(".")[0].split("___")[1]
        key = video_path.split("/")[-2] + "___" + start + "___" + end
        save_path = os.path.join(args_cli.data_save_dir, "pose_tokens", "20", f"{sample_name}", f"{start}___{end}.npz")


        model.eval() 
        ds_inf = MotionInferenceDataset(
            pt_path=args.data_dir,
            key=key,  # ←指定したいkey
            clip_len=21,
            overlap=2,
        )
        dl = DataLoader(
            ds_inf,
            batch_size=1,
            # shuffle=True,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_stack,   # ★ここ変更
        )


        idx_all = []
        for batch in dl:
            recon, losses, idx = model(batch["mB"].to(device), batch["mH"].to(device))
            idxH = idx["idxH"].detach().cpu().numpy()
            idxB = idx["idxB"].detach().cpu().numpy()
            idx = np.concatenate([idxH, idxB], axis=-1)
            idx = idx.reshape(-1)
            idx_all.append(idx)
        idx_all = np.array(idx_all)
        print(f"shape: {idx_all.shape}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, idx_all)


if __name__ == "__main__":
    main()