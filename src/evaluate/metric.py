import torch
@torch.no_grad()
def codebook_stats(idx: torch.Tensor, K: int):
    flat = idx.reshape(-1)
    hist = torch.bincount(flat, minlength=K).float()
    p = hist / (hist.sum() + 1e-8)
    usage = (hist > 0).float().mean().item()
    perplexity = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    return usage, perplexity

def mpjpe(pred, gt, root_align=True):
    """
    pred, gt: (B,T,J,3)
    """
    if root_align:
        pred = pred - pred[..., :1, :]
        gt = gt - gt[..., :1, :]
    return torch.norm(pred - gt, dim=-1).mean()