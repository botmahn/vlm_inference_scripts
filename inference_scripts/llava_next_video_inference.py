import os
import json
import argparse
from tqdm import tqdm

import torch
import numpy as np
import av
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def process_video(video_path, processor, model, output_dir, user_prompt):
    video_id = os.path.basename(video_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "video"},
            ],
        },
    ]

    try:
        container = av.open(video_path)
    except av.AVError:
        print(f"Could not open video: {video_id}")
        return

    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        print(f"No frames in video: {video_id}")
        return

    indices = np.linspace(0, total_frames - 1, num=8, dtype=int)
    clip = read_video_pyav(container, indices)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    caption = processor.decode(output[0][2:], skip_special_tokens=True)

    result = {
        "video_name": video_id,
        "prompt": user_prompt,
        "caption": caption
    }

    out_path = os.path.join(output_dir, f"{os.path.splitext(video_id)[0]}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)


def main(video_folder, output_dir, model_id, user_prompt):
    os.makedirs(output_dir, exist_ok=True)

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True
    )
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_folder, video_file)
        process_video(video_path, processor, model, output_dir, user_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video captioning with LLaVA-NeXT.")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder to save JSON files")
    parser.add_argument("--model_id", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Describe the video in detail.", help="Prompt to send to the model")

    args = parser.parse_args()
    main(args.video_folder, args.output_dir, args.model_id, args.prompt)
