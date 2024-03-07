import os
import argparse
from einops import rearrange
import random
import numpy as np
from PIL import Image
import transformers

import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from rovi.models.unet import RoviModel
from rovi.pipelines.pipeline_rovi_mllm import RoviPipelineMLLM
from rovi.util import save_videos_grid

from rovi.llm.llava.model.language_model.llava_llama_inpaint import LlavaLlamaInpaint
from rovi.llm.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from rovi.llm.llava.conversation import conv_templates, SeparatorStyle
from rovi.llm.llava.model.builder import load_pretrained_model
from rovi.llm.llava.utils import disable_torch_init, load_mllm_weights
from rovi.llm.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, get_prompt_id_start
from rovi.llm.llava.train.train import ModelArguments


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

def _clip_transform(n_px=336):
    return transforms.Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # clip-vit-large-p14-336
    ])

def transform_first_frame(video_path):
    first_frame_name = list(sorted(os.listdir(video_path)))[0]
    image = Image.open(os.path.join(video_path, first_frame_name)).convert("RGB")
    image = _clip_transform(336)(image)
    return image


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
    pipe = RoviPipelineMLLM.from_pretrained(args.ckpt_path, unet=unet, torch_dtype=weight_dtype).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./checkpoints/llava-v1.5-7b",
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    bnb_model_from_pretrained_args = {}
    model_llm = LlavaLlamaInpaint.from_pretrained(
        "./checkpoints/llava-v1.5-7b",
        mm_vision_tower="./checkpoints/clip-vit-large-patch14-336",
        # use_flash_attention_2=False,
        torch_dtype=weight_dtype,
        **bnb_model_from_pretrained_args
    )
    model_args = ModelArguments(
        vision_tower="./checkpoints/clip-vit-large-patch14-336",
        pretrain_mm_mlp_adapter=None,
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,)
    model_llm.get_model().initialize_vision_modules(
        model_args=model_args,
    )
    load_mllm_weights(model_llm, os.path.join(args.ckpt_path, f'llava', 'llava_weights.pth'))
    model_llm.to(device).to(weight_dtype)

    pixel_values = load_video(args.video_path)
    batch_size, video_length, num_channels, height, width = pixel_values.shape

    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    condition_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    condition_latents = rearrange(condition_latents, "(b f) c h w -> b c f h w", f=video_length)
    condition_latents = condition_latents * 0.18215

    question = args.request
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates["v1_inpaint"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image_tensor = transform_first_frame(args.video_path).to(model_llm.dtype).cuda()
    print(image_tensor.shape)

    with torch.inference_mode():
        output_dict = model_llm.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).cuda(),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True)
        
        hidden_states = output_dict.hidden_states

        transfer_feature = []
        transfer_feature.append(hidden_states[0][-1][:,-1:,:])
        for embeds in hidden_states[1:]:
            embed = embeds[-1]
            transfer_feature.append(embed)
        transfer_feature = torch.cat(transfer_feature, dim=1)
        transfer_feature = model_llm.transfer_head(transfer_feature)

    output_ids = output_dict.sequences
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    output_ids = output_ids[:, input_token_len:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    print(f"\n\n{outputs}\n\n")

    prompt_id_start = get_prompt_id_start(outputs, tokenizer, IMAGE_TOKEN_INDEX)
    if prompt_id_start > transfer_feature.shape[-2]:
        sp_transfer_feature = transfer_feature
    else:
        sp_transfer_feature = transfer_feature[:,prompt_id_start:,:]
    
    guidance_scale = args.gs
    image_guidance_scale = args.igs
    num_inference_steps = 50

    pipe_output = pipe(sp_transfer_feature, img_condition=condition_latents, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, generator=generator)
    
    video = pipe_output.videos

    # save fist 100 charaters of the request to avoid too long file names
    output_path = f"./results/{args.request[:100]}.gif"
    save_videos_grid(video, output_path)

if __name__ == '__main__':
    """
    srun --partition=s1_mm_research --job-name=layout_multi --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --kill-on-bad-exit=1 --quotatype=auto \
    -x SH-IDC1-10-140-0-203 \
    nohup python -m inference_interactive \
        --request "I have this incredible shot of a pelican gliding in the sky, but there's another bird also captured in the frame. Can you help me make the picture solely about the pelican?" \
        > nohup.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path of the video folder', default='videos/city-bird')
    parser.add_argument('--ckpt_path', help='path of the checkpoint folder', default='./checkpoints/lgvi-i')
    parser.add_argument('--request', help='referring expression', default="I have this incredible shot of a pelican gliding in the sky, but there's another bird also captured in the frame. Can you help me make the picture solely about the pelican?")
    parser.add_argument('--gs', type=float, help='language guidance scale', default=3.0)
    parser.add_argument('--igs', type=float, help='image guidance scale', default=1.5)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    main(args)