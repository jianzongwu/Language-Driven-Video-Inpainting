import os
import imageio
import math
import numpy as np

import torch
import torchvision
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=1000 * (1 / fps), loop=0)

def get_unet_constant_deocder_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch=-1):
    def lr_lambda_unet(current_step: int):
        return 1
    
    def lr_lambda_decoder(current_step: int):
        if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda=[lr_lambda_unet, lr_lambda_decoder])

def get_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch=-1):
    def lr_lambda_cosine(current_step: int):
        if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda_cosine)

def get_mask_loss_weight_scale_quadratic(timesteps, total_steps=1000):
    # use quadratic function as mask loss weight scale
    scales = []
    for t in timesteps:
        scale = t * (total_steps - t) * 4.0 / total_steps**2
        scales.append(scale)
    return scales
