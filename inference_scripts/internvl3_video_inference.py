import os
import argparse
import json
import math
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from threading import Thread
from transformers import AutoModel, AutoTokenizer, AutoConfig, TextIteratorStreamer
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )) for i in range(blocks)
    ]
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    for module in [
        'vision_model', 'mlp1', 'language_model.model.tok_embeddings',
        'language_model.model.embed_tokens', 'language_model.output',
        'language_model.model.norm', 'language_model.model.rotary_emb',
        'language_model.lm_head']:
        device_map[module] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    transform = build_transform(input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    pixel_values_list, num_patches_list = [], []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img_tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    device_map = split_model(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

    video_files = [f for f in os.listdir(args.video_folder) if f.endswith('.mp4')]
    for video_file in tqdm(video_files, desc="Processing Videos"):
        basename = video_file.split(".mp4")[0]
        video_path = os.path.join(args.video_folder, video_file)
        pixel_values, num_patches_list = load_video(
            video_path,
            num_segments=args.num_frames,
            max_num=args.patch_size
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + args.prompt

        if args.stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
            generation_config = dict(max_new_tokens=1024, do_sample=True, streamer=streamer)
            thread = Thread(target=model.chat, kwargs=dict(
                tokenizer=tokenizer, pixel_values=pixel_values, question=question,
                history=None, return_history=False, generation_config=generation_config
            ))
            thread.start()
            response = ""
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response += new_text
        else:
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response = model.chat(
                tokenizer, pixel_values, question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            print(f'User: {question}\nAssistant: {response}')

        with open(f"{args.outdir}/{basename}.json", 'w') as outfile:
            json.dump({
                'video': basename + ".mp4",
                'prompt': args.prompt,
                'response': response
            }, outfile, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InternVL3 Video Captioning Script.")
    parser.add_argument('--video_folder', type=str, required=True, help='Path to folder with .mp4 videos')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-9B', help='Model path')
    parser.add_argument('--prompt', type=str, default='Describe the video in detail.', help='Text prompt')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to sample')
    parser.add_argument('--patch_size', type=int, default=3, help='Max number of tiles per frame')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save responses')
    parser.add_argument('--stream', action='store_true', help='Enable streaming output mode')
    args = parser.parse_args()
    main(args)
