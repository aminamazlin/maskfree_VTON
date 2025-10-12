import cv2
import numpy as np
import supervision as sv
import json
import torch
import torchvision
from tqdm import tqdm
import random
import sys
from pathlib import Path
import os
import os.path as osp
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

query_prompt = \
    '''
    Analyze these two images of the same model wearing different clothes. Check if each image shows a necklace. 
    If both do, check if it's the exact same necklace. Return ONLY valid JSON in this format:

    {
    "image_0_has_necklace": boolean,
    "image_1_has_necklace": boolean,
    "both_have_necklaces": boolean,
    "same_necklace": "yes" | "no" | "unknown",
    "confidence": number,
    "notes": string
    }

    '''

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

if _name_ == '_main_':
    json_file = "infer_necklace_objects.json"

    # Load existing data or create empty dict if file doesn't exist or is empty
    try:
        with open(json_file, 'r') as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
            else:
                data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
        print(f"Creating new {json_file} file")

    # get the original images
    orig_tryon_dir = Path("data/orig_tryon")
    orig_images = sorted(orig_tryon_dir.glob("*.jpg"))

    new_tryon_dir = Path("data/new_tryon")

    for orig_img in tqdm(orig_images, desc="Analyze necklaces.."):
        try:
            # get the name of the original image
            name = orig_img.name

            # construct the new tryon path and load 
            new_tryon_img_path = new_tryon_dir / name

            orig_img_path = orig_tryon_dir / name
            
            # Check if both images exist
            if not orig_img_path.exists():
                print(f"Original image not found: {orig_img_path}")
                continue
                
            if not new_tryon_img_path.exists():
                print(f"New tryon image not found: {new_tryon_img_path}")
                continue


            # pass thorugh qwen image edit
            messages = [
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image",
                            "image": str(orig_img_path)
                        },
                        {
                            "type":"image",
                            "image": str(new_tryon_img_path)
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
            
            # Parse the JSON output
            try:
                # Get the first output and try to parse as JSON
                raw_output = output_text[0].strip()
                
                # Try to extract JSON if it's wrapped in other text
                json_start = raw_output.find('{')
                json_end = raw_output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_output[json_start:json_end]
                    # convert json to python dict
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in output")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse JSON for {name}: {raw_output}")
                # Create a fallback result
                result = {
                    "image_0_has_necklace": False,
                    "image_1_has_necklace": False,
                    "both_have_necklaces": False,
                    "same_necklace": "unknown",
                    "confidence": 0.0,
                    "notes": f"JSON parsing failed: {str(e)}"
                }
            
            # Add the result to our data with the image name as key
            data[name] = result
            print(f"Processed {name}: {result}")
            
        except Exception as e:
            print(f"Error processing {orig_img.name}: {str(e)}")
            # Add an error entry for this image
            data[orig_img.name] = {
                "image_0_has_necklace": False,
                "image_1_has_necklace": False,
                "both_have_necklaces": False,
                "same_necklace": "unknown",
                "confidence": 0.0,
                "notes": f"Processing error: {str(e)}"
            }
            continue
    
    # Save the updated data back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {json_file}")