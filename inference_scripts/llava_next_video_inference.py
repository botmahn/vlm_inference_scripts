import os
import argparse
import json
import av
import torch
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def main(args):
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto'
    )

    processor = LlavaNextVideoProcessor.from_pretrained(args.model)

    default_prompt = (
        "You are analyzing a dashcam video from a vehicle driving through traffic. "
        "Your task is to generate a detailed, factual caption that accurately describes: "
        "1. Road types (highway, urban street, rural road, intersection, etc.), "
        "2. Infrastructure visible (traffic lights, signs, lane markings, barriers, bridges, tunnels), "
        "3. Traffic density (heavy, moderate, light, or none), "
        "4. Road participants (cars, trucks, pedestrians, cyclists, motorcyclists), "
        "5. Environmental conditions (weather, time of day, visibility), "
        "6. Traffic events (stopping, turning, merging, etc.), "
        "7. Crossings or junctions (pedestrian crossings, intersections, highway exits). "
        "Important guidelines: Only describe what is clearly visible in the video. Do not hallucinate or make assumptions about elements not visible. "
        "Be specific about positions, use objective language without speculation, maintain a consistent tense, avoid mentioning uncertain elements. "
        "Focus on the most prominent and relevant features of the traffic scene."
    )

    prompt_text = args.prompt if args.prompt.strip() else default_prompt

    os.makedirs(args.outdir, exist_ok=True)
    video_files = [f for f in os.listdir(args.video_folder) if f.endswith(".mp4")]

    for video_file in tqdm(video_files, desc="Processing Videos"):
        basename = os.path.splitext(os.path.basename(video_file))[0]
        video_path = os.path.join(args.video_folder, video_file)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video"},
                ],
            },
        ]

        chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
        clip = read_video_pyav(container, indices)  # (N, H, W, 3)

        if clip.shape[0] != args.num_frames:
            print(f"Warning: Extracted {clip.shape[0]} frames instead of {args.num_frames}.")

        # Convert to float32 for processor
        clip = torch.tensor(clip).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W)

        inputs = processor(text=chat_prompt, videos=[clip], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        output_text = processor.decode(output_tokens[0][2:], skip_special_tokens=True)

        print(f"\nVideo: {video_file}\nResponse: {output_text}\n")

        result = {
            "video": video_file,
            "model": args.model,
            "num_frames": args.num_frames,
            "max_tokens": args.max_new_tokens,
            "prompt": prompt_text,
            "response": output_text,
        }

        out_path = os.path.join(args.outdir, f"{basename}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved: {out_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLaVA-Next Video Captioning Script")
    parser.add_argument('--video_folder', type=str, required=True, help='Path to folder with .mp4 videos')
    parser.add_argument('--model', type=str, default='llava-hf/LLaVA-NeXT-Video-7B-hf', help='Model identifier or path')
    parser.add_argument('--prompt', type=str, default='', help='Optional text prompt override')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to sample per video')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory to save JSON results')
    args = parser.parse_args()
    main(args)
