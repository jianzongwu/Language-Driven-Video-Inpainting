import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.dataset == 'rovi':
        data_root = 'data/rovi_dataset'
    elif args.dataset == 'gqa':
        data_root = 'data/gqa_inpaint'
    
    json_path = f"{data_root}/{args.split}_itr.json"
    output_json_path = f"{data_root}/{args.split}_res.json"

    with open(json_path, 'r') as f:
        data_infos = json.load(f)

    with open(output_json_path, 'r') as f:
        output_data_infos = json.load(f)

    # question = 'describe this image'
    # question = 'Can you remove the animal in this picture?'# Please give the prompt'
    for video_name in tqdm(list(data_infos.keys())):
        for obj_id, values in data_infos[video_name]['objects'].items():
            repeat = 0
            # while output_data_infos[video_name]['objects'][obj_id].get('response', None) is None:
            frame = np.random.choice(list(os.listdir(os.path.join(data_root, 'JPEGImages', video_name))))
            image_path = os.path.join(data_root, 'JPEGImages', video_name, frame)
            question = values['requests'][0]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(image_path)
            # (3, H, W)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # print(outputs)
            if "{{prompt}}" not in outputs or "{{/prompt}}" not in outputs:
                repeat += 1
                print(video_name, obj_id, frame, repeat, "llava did not generate {{prompt}} and {{/prompt}}")
                if repeat >= 6:
                    print("repeat in this object >= 6 times, skip this object")
                    break
            else:
                output_data_infos[video_name]['objects'][obj_id]['response'] = [outputs]

    with open(output_json_path, 'w') as f:
        json.dump(output_data_infos, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="v1_inpaint")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--dataset', type=str, help='rovi/gqa', default='rovi')
    parser.add_argument('--split', type=str, help='train/test', default='test')
    args = parser.parse_args()

    main(args)
