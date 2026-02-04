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
    ap.add_argument("--config", type=str, default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/config.yaml")
    ap.add_argument("--ckpt", type=str, default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args_cli = ap.parse_args()

    video_base_dir = os.path.join(os.getenv("DATA_ROOT"), args_cli.split, "takes_clipped", "egoexo", "videos")
    data_save_dir = os.path.join(os.getenv("DATA_ROOT"), args_cli.split, "takes_clipped", "egoexo")
    human_pose_dir = os.path.join(os.getenv("DATA_ROOT"), "ee4d", "ee4d_motion_uniegomotion", "uniegomotion", "ee_train_joints_tips.pt")


    args = OmegaConf.load(args_cli.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(args, device)
    ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    video_paths = glob.glob(os.path.join(video_base_dir, "**/*.mp4"), recursive=True)
    video_paths.sort()
    
    j = 0

    for video_path in tqdm(video_paths, desc="Processing videos"):
        sample_name = video_path.split("/")[-2]
        start = video_path.split("/")[-1].split(".")[0].split("___")[0]
        end = video_path.split("/")[-1].split(".")[0].split("___")[1]
        key = video_path.split("/")[-2] + "___" + start + "___" + end
        # save_path = os.path.join(args_cli.data_save_dir, "pose_tokens", "20", f"{sample_name}", f"{start}___{end}.npz")
        save_dir = os.path.join(data_save_dir, "tok_pose",  f"{sample_name}")
        os.makedirs(save_dir, exist_ok=True)


        model.eval() 
        ds_inf = MotionInferenceDataset(
            pt_path=human_pose_dir,
            key=key,  # ←指定したいkey
            clip_len=20,
            overlap=1,
            include_fingertips=args.include_fingertips,

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
        i = 0
        for batch in dl:
            save_path = os.path.join(save_dir, f"{start}___{end}_{i}.npz")
            if os.path.exists(save_path) and not args_cli.overwrite:
                continue
            recon, losses, idx = model(batch["mB"].to(device), batch["mH"].to(device))

            idxH = idx["idxH"].detach().cpu().numpy()
            idxB = idx["idxB"].detach().cpu().numpy()
            idx = np.concatenate([idxH, idxB], axis=-1)
            if j == 0 and i == 0:
                print(f"idx shape: {idx.shape} idxB shape: {idxB.shape} idxH shape: {idxH.shape}")
            idx = idx.reshape(-1)
            np.savez_compressed(save_path, idx=idx)
            i += 1
        j += 1

if __name__ == "__main__":
    main()