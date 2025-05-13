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
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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

def main(args):

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            args.model, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map='auto'
    )

    processor = LlavaNextVideoProcessor.from_pretrained(args.model)

    video_files = os.listdir(args.video_folder)
    for video_file in tqdm(video_files, desc="Processing Videos"):
        basename = os.path.basename(video_file).split(".mp4")[0]
        video_file = os.path.join(args.video_folder, video_file)

        conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "video"},
                    ],
                },
            ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        container = av.open(video_file)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
        clip = read_video_pyav(container, indices)
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

        token_output = model.generate(**inputs_video, max_new_tokens=args.max_new_tokens, do_sample=False)
        output = processor.decode(token_output[0][2:], skip_special_tokens=True)

        data = {
            "video": basename + ".mp4",
            "prompt": args.prompt,
            "response": output,
        }

        os.makedirs(args.outdir, exist_ok=True)
        with open(f"{args.outdir}/{basename}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)
            print(f"Saved: {args.outdir}/{basename}.json\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InternVL3 Video Captioning Script.")
    parser.add_argument('--video_folder', type=str, required=True, help='Path to folder with .mp4 videos')
    parser.add_argument('--model', type=str, default='llava-hf/LLaVA-NeXT-Video-7B-hf', help='Model path')
    parser.add_argument('--prompt', type=str, default='Describe the video in detail.', help='Text prompt')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to sample')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max number of generated tokens.')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save responses')
    args = parser.parse_args()
    main(args)
