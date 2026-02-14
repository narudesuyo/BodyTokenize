import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# --- Path Setup ---
sys.path.append(".")

# --- User Imports ---
from src.train.utils import build_model_from_args
from src.dataset.collate import collate_stack
# Import the new dataset class
from src.dataset.infer_all_loder import MotionAllInferenceDataset 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.pt")
    ap.add_argument("--pt_path", type=str, default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_tips.pt", help="Input motion .pt file")
    ap.add_argument("--save_root", type=str, default="/large/naru/EgoHand/data/train/takes_clipped/egoexo/tok_pose/all_new", help="Output root dir")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    
    args_cli = ap.parse_args()

    # 1. Config & Model
    print("Loading Model...")
    args = OmegaConf.load(args_cli.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model_from_args(args, device)
    ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    
    # Handle DDP state_dict keys if needed
    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    # 2. Dataset & DataLoader
    print("Initializing Dataset...")
    ds = MotionAllInferenceDataset(
        pt_path=args_cli.pt_path,
        clip_len=20,     # Match your training clip length
        overlap=1,
        include_fingertips=args.include_fingertips,
        base_idx=args.base_idx,
    )

    dl = DataLoader(
        ds,
        batch_size=args_cli.batch_size,
        shuffle=False, 
        num_workers=args_cli.num_workers,
        pin_memory=True,
        drop_last=False,
        # collate_fn=collate_stack 
    )

    # 3. Inference Loop
    print(f"Start Inference on {len(ds)} clips...")
    os.makedirs(args_cli.save_root, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Processing"):
            mB = batch["mB"].to(device, non_blocking=True)
            mH = batch["mH"].to(device, non_blocking=True)
            
            # Metadata for saving
            keys = batch["key"]          # list
            save_idxs = batch["save_idx"] # tensor/list

            # Inference
            _, _, code_indices = model(mB, mH)

            idxH = code_indices["idxH"].cpu().numpy() # (B, T)
            idxB = code_indices["idxB"].cpu().numpy() # (B, T)
            
            # Save results
            B_curr = idxH.shape[0]
            for i in range(B_curr):
                k = keys[i]
                file_idx = save_idxs[i]
                
                # Create directory for key
                seq_name = k.split("___")[0]
                key_save_dir = os.path.join(args_cli.save_root, seq_name)
                os.makedirs(key_save_dir, exist_ok=True)
                
                save_path = os.path.join(key_save_dir, k.split("___")[1]+"___"+k.split("___")[2]+"_"+f"{file_idx:04d}.npz")
                
                if os.path.exists(save_path) and not args_cli.overwrite:
                    continue

                # Stack tokens: Hand, Body -> (T, 2) -> flatten
                token_cat = np.stack([idxB[i], idxH[i]], axis=-1)
                np.savez_compressed(save_path, token_cat)

    print("Done.")

if __name__ == "__main__":
    main()