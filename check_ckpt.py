import torch
ckpt_path = "/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt"
ckpt = torch.load(ckpt_path, weights_only=False)
print(ckpt.keys())