import os
import argparse
from einops import rearrange
import random
import numpy as np
from PIL import Image

import torch

from rovi.models.unet import RoviModel
from rovi.pipelines.pipeline_rovi import RoviPipeline
from rovi.util import save_videos_grid

WIDTH = 512
HEIGHT = 320
device = torch.device("cuda:0")

def load_video(video_path):
    frames = []
    max_num_frames = 24
    frame_files = list(sorted(os.listdir(video_path)))[:max_num_frames]
    for frame_name in frame_files:
        image = Image.open(os.path.join(video_path, frame_name)).convert("RGB")
        image = image.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)
        frames.append(image)
    frames = np.stack(frames, axis=2)
    frames = torch.from_numpy(frames).permute(2, 3, 0, 1).contiguous().unsqueeze(0)
    frames = frames.float().div(255).clamp(0, 1).half().cuda() * 2.0 - 1.0
    return frames

@torch.no_grad()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    generator = torch.Generator(device)
    generator.manual_seed(args.seed)

    weight_dtype = torch.float16
    unet = RoviModel.from_pretrained(args.ckpt_path, subfolder='unet')
    unet.to(device).to(weight_dtype)
    pipe = RoviPipeline.from_pretrained(args.ckpt_path, unet=unet, torch_dtype=weight_dtype).to(device)

    pixel_values = load_video(args.video_path)

    batch_size, video_length, num_channels, height, width = pixel_values.shape

    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    condition_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    condition_latents = rearrange(condition_latents, "(b f) c h w -> b c f h w", f=video_length)
    condition_latents = condition_latents * 0.18215
    
    guidance_scale = args.gs
    image_guidance_scale = args.igs
    num_inference_steps = 50

    pipe_output = pipe(args.expr, img_condition=condition_latents, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, generator=generator)
    
    video = pipe_output.videos

    output_path = f"./results/{args.expr}.gif"
    save_videos_grid(video, output_path)

if __name__ == '__main__':
    """
    srun --partition=s1_mm_research --job-name=layout_multi --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --kill-on-bad-exit=1 --quotatype=auto \
    nohup python -m inference_referring \
        --expr "remove the bird on left" \
        > nohup.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path of the video folder', default='videos/two-birds')
    parser.add_argument('--ckpt_path', help='path of the checkpoint folder', default='./checkpoints/lgvi')
    parser.add_argument('--expr', help='referring expression', default='remove the bird on left')
    parser.add_argument('--gs', type=float, help='language guidance scale', default=3.0)
    parser.add_argument('--igs', type=float, help='image guidance scale', default=1.5)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    main(args)