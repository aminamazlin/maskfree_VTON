from typing import List, Union, Dict
from tqdm import tqdm
import json
from diffusers import StableDiffusionPipeline
import torch
from typing import List
from PIL import Image
import pathlib
import urllib.parse
import os
import mlflow


def _iterate_items(eval_data: Union[List, Dict]):
    for entry in eval_data:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Unexpected entry format: {entry}")
        
        garment_class = list(entry.keys())[0]             # "full_body"
        item = dict(entry[garment_class])                 # get the inner dict safely
        item.setdefault("class_name", garment_class)      # ensure there's a class name key

        yield garment_class, item

def get_local_artifact_dir(artifact_subdir: str = ""):
    """
    Returns a *local* filesystem path inside the current run's artifact store.
    Works when MLflow artifact URI is file://...
    """

    # Gets the full artifact URI (like file:///home/user/mlruns/...) from MLflow for the given subdirectory.
    uri = mlflow.get_artifact_uri(artifact_subdir)
    parsed = urllib.parse.urlparse(uri)

    assert parsed.scheme == "file", f"Artifact store is not local: {uri}"

    return pathlib.Path(urllib.parse.unquote(parsed.path))


def evaluate_lora(
    base_model_id_or_path: str,
    lora_dir: str,
    eval_json: Union[str, List, Dict],
    tryon_subdir: str = "tryon", 
    artifact_root_dir: str = "stage1_eval",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    # Load evalulation data
    with open(eval_json, "r") as file:
        eval_data = json.load(file)

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id_or_path,
        torch_dtype=torch.float16
    ).to(device)
    # Load LoRa weights
    pipe.load_lora_weights(lora_dir)

    for garment_class, item in tqdm(list(_iterate_items(eval_data)), desc="Generating stage 1 try-on images"):
        # Get artifact dir
        tryon_root = get_local_artifact_dir(f"{artifact_root_dir}/{tryon_subdir}")

        # Load prompt
        prompt = item["tryon_prompt"]

        # Load traceless erased image
        tryon_in_path = item["input_tryon_img_path"]
        input_img = Image.open(tryon_in_path ).convert("RGB")

        # Get base name
        base_name = os.path.splitext(os.path.basename(tryon_in_path))[0]

        # Load zero mask
        w, h = input_img.size
        zero_mask = Image.new("L", (w, h), 0)

        # Inference
        out = pipe(
            prompt=prompt,
            image=input_img,
            mask=zero_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        # Save under artifacts/stage1_eval/tryon/<garment_class>/<base>.png
        save_dir = tryon_root/ garment_class
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{base_name}.png"
        out.save(out_path)

        # Update JSON with absolute path that metric code can open directly
        item["output_tryon_img_path"] = str(out_path)
        
    # Write updated JSON into artifacts
    json_dir = get_local_artifact_dir(artifact_root_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = json_dir / eval_json
    with open(out_json_path, "w") as file:
        json.dump(eval_data, file, indent=2)
    
    return str(out_json_path)
