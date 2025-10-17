import cv2
import numpy as np
import supervision as sv
import json
from typing import Union, List, Dict
import torch
import torchvision
from tqdm import tqdm
import random
import sys
import os
from pathlib import Path
import os.path as osp
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


device = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_PROMPT = """


    Generate a clear, factual description of the given image.
    Only describe what is directly visible — the model, what the model is wearing, the pose, the background setting, and any additional objects.
    Be specific about the clothing, shoes, accessories, hairstyle, and any bags or jewelry visible on the model.

    Do not describe or speculate about lighting, brightness, shadows, reflections, materials, textures, atmosphere, mood, time of day, or photo quality.
    Do not add interpretations such as scientific, artistic, or emotional meanings.

    All added objects in the image are floating near the model and are never held, touched, or interacted with. 
    Always describe clearly where the object is floating (for example: beside the model, above the shoulder, near the feet, or in front of the body).


    Describe the image in about 65 words.
    Focus on concrete visual details: the model’s clothing, posture, surroundings, and visible floating objects.
    Emphasize what the model is wearing — describe the {garment_class}, other clothing items, footwear, hairstyle, accessories, and any visible bags or jewelry.
    Include any surrounding objects naturally, but avoid mentioning that the model is holding the object, even if it seems like it. 
    The main focus should remain on describing the model and what the model is wearing.

    Examples outputs:
    - "A young woman standing in a studio with a white background. 
    She is wearing a denim dress with a button-down collar and long sleeves. 
    The dress is knee-length and falls above her knees. The woman is also wearing black ankle boots with a pointed toe and a low heel. 
    She has a brown crossbody bag with a strap across her shoulder. A big shovel is floating beside her. 
    The overall style of the outfit is casual and minimalistic."

    - "A young woman standing in a bright laboratory filled with computers, equipment, and cables. 
    Her hair is styled in loose waves and she is wearing a fitted dress with swirls of purple and gold and long sleeves. 
    She has small hoop earrings and her hands are placed on her hips. 
    In front of her, a small wooden boat rests against her legs."

    - "A woman standing on a snowy forest path, dawn mist curling between tall pine trees. 
    She is wearing a flowing white lace dress with a V-neck and flared skirt. 
    She holds a fuzzy white handbag and wears golden hoop earrings and matching heels. 
    A giant steaming coffee cup floats beside her."
    """

# Init vision model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Init vision processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def _iterate_items(eval_data: dict):
    """
    Yields (class_name, node) where:
      class_name -> full_body, lower_body, upper_body, outerwear
      node       -> dict with keys: "person", "object", "tryon_prompt" 
    Skips empty/invalid entries gracefully.
    """
    for garment_class, items_list in eval_data.items():
        if not isinstance(items_list, list):
            print(f"[WARN] Skipping non-list items for {garment_class}: {items_list}")
            continue
            
        for node in items_list:
            if not isinstance(node, dict):
                print(f"[WARN] Skipping non-dict node for {garment_class}: {node}")
                continue
            
            person = node.get("person", {})
            obj = node.get("object", {})

            if not person or not obj:
                print(f"[WARN] Missing 'person' or 'object' for class '{garment_class}'")
                continue
            
            required_person = ["input_tryon_img_path"]
            required_object = ["img_path"]

            if any(k not in person or not person[k] for k in required_person):
                print(f"[WARN] Incomplete 'person' fields for '{garment_class}'")
                continue
            if any(k not in obj or not obj[k] for k in required_object):
                print(f"[WARN] Incomplete 'object' fields for '{garment_class}'")
                continue

            yield garment_class, node


if __name__ == '__main__':
    data_dir = Path("data")
    # json_file = "tryon_data.json"
    json_path = data_dir / "tryon_data.json"

    # Load data
    try:
        data_str = json_path.read_text(encoding="utf-8").strip()
        data = json.loads(data_str) if data_str else []
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"JSON file not found or invalid at: {json_path}")
        sys.exit(1)

    for garment_class, item in tqdm(list(_iterate_items(data)), desc="Generating tryon descriptions for stage 1"):
        try:
            person = item["person"]
            object_info = item["object"]

            tryon_img_path = Path(person["input_tryon_img_path"])
            garment_img_path = Path(object_info["img_path"]) 
                      
            if not tryon_img_path.exists() or not garment_img_path.exists():
                print(f"Try-on image or garment image not found: {tryon_img_path}, {garment_img_path}")
                person["tryon_description"] = "unknown"
                continue

            # object_category = object_info.get("object_category", garment_class)
            query_prompt = QUERY_PROMPT.format(garment_class="dress")

            # pass thorugh qwen vision model
            messages = [
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image",
                            "image": str(tryon_img_path)
                        },
                        {
                            "type":"text",
                            "text": query_prompt
                        }
                    ]
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            person["tryon_description"] = output_text
            print(f"Desc for {tryon_img_path}: {output_text}")
      

        except Exception as e:
            print(f"Error processing: {tryon_img_path}. Error: {e}")
            person["tryon_description"] = "unknown"
            continue

    # Save
    try:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Failed to save results to {json_path}: {e}")
        sys.exit(1)