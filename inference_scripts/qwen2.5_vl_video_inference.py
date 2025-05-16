import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Load model and processor once
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    video_files = [f for f in os.listdir(args.video_folder) if f.endswith(".mp4")]

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

    prompt = args.prompt if args.prompt.strip() else default_prompt

    for video_file in tqdm(video_files, desc="Processing videos"):
        basename = os.path.splitext(os.path.basename(video_file))[0]
        video_path = os.path.join(args.video_folder, video_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": args.max_pixels[0] * args.max_pixels[1],
                        "fps": args.fps,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=args.fps,
            padding=True,
            return_tensors="pt",
            #**video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"[{basename}] Response:", output_text)

        # Save output
        output_data = {
            'model': args.model,
            'fps': args.fps,
            'prompt': prompt,
            'response': output_text,
        }
        outpath = os.path.join(args.outdir, f"{basename}.json")
        with open(outpath, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True, help="Path to folder containing MP4 videos.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save output JSON files.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path.")
    parser.add_argument("--prompt", type=str, default='', help="Prompt to ask the model.")
    parser.add_argument("--fps", type=int, default=1.0, help="Frames per second to extract.")
    parser.add_argument("--max_pixels", type=list, default=[448,448], help="Max pixels to process.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate.")
    args = parser.parse_args()
    main(args)
