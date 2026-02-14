import torch
def collate_stack(batch):
    # default_collateでも良いけど、keyとか文字列が混じる場合に安全にしたいならこれ
    body = torch.stack([b["body"] for b in batch], dim=0)  # (B,T,263)
    hand = torch.stack([b["hand"] for b in batch], dim=0)  # (B,T,360)
    out = {"mB": body, "mH": hand}
    if "kp52" in batch[0]:
        out["kp52"] = torch.stack([b["kp52"] for b in batch], dim=0)  # (B,T,52,3)
    # 追加で必要なら
    if "key" in batch[0]:
        out["keys"] = [b["key"] for b in batch]
    if "start" in batch[0]:
        out["start"] = torch.tensor([b["start"] for b in batch], dtype=torch.long)
    if "end" in batch[0]:
        out["end"] = torch.tensor([b["end"] for b in batch], dtype=torch.long)
    if "Tfull" in batch[0]:
        out["Tfull"] = torch.tensor([b["Tfull"] for b in batch], dtype=torch.long)
    if "clip_index" in batch[0]:
        out["clip_index"] = torch.tensor([b["clip_index"] for b in batch], dtype=torch.long)
    return out