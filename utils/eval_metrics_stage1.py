from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from transformers import pipeline, CLIPProcessor, CLIPModel
import torch
import torchvision
import torchvision.transforms as T
import supervision as sv
from PIL import Image
import numpy as np
from typing import List
import json
import tqdm
import os
import logging
import mlflow

# Create a basic logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s: %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "Grounded-Segment-Anything/checkpoints/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "Grounded-Segment-Anything/checkpoints/sam_vit_h_4b8939.pth"

# CLIP checkpoint
CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"

# CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# GroundingDINO model
grounding_dino_model = Model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH )

# SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8



to_pil = T.ToPILImage()

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
            
            required_person = ["input_tryon_img_path", "tryon_description"]

            if any(k not in person or not person[k] for k in required_person):
                print(f"[WARN] Incomplete 'person' fields for '{garment_class}'")
                continue

            yield garment_class, node

def detect(
    image: np.ndarray, 
    classes: List[str],
    box_threshold: float
):

    # Detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=TEXT_THRESHOLD
    )

    if len(detections.xyxy) > 0:
        # Perform non max supression
        nms_indices = torchvision.ops.nms(
            boxes=torch.from_numpy(detections.xyxy),
            scores=torch.from_numpy(detections.confidence),
            iou_threshold= NMS_THRESHOLD
        ).numpy().tolist()

        # Keep only filtered boxes
        detections.xyxy = detections.xyxy[nms_indices]
        detections.confidence = detections.confidence[nms_indices]
        detections.class_id = detections.class_id[nms_indices]

        if 'earrings' in classes:
            topk = 2
        else:
            topk = 1

        # Keep only topK
        detections.xyxy = detections.xyxy[:topk]
        detections.confidence = detections.confidence[:topk]
        detections.class_id = detections.class_id[:topk]
        return detections

    return detections


def segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)

    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        # Select the highest score mask
        index = np.argmax(scores)
        mask = masks[index]
        result_masks.append(mask)

    # Return a list of binary masks
    return np.array(result_masks)


def generate_mask(img_path, prompt, box_threshold=BOX_THRESHOLD):
    classes = [prompt]

    # Convert image to numpy array
    image = np.array(Image.open(img_path).convert("RGB"))

    # Detect
    detections = detect(image=image, classes=classes, box_threshold=box_threshold)

    # If no detections, return empty mask + original image
    if len(detections.xyxy) == 0:
        empty = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.uint8)
        return empty, image

    # Convert detections to masks and add to the Detections object
    detections.mask = segment(image=image, xyxy=detections.xyxy)

    # Annonate mask
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections
    ]

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Merge all object masks into one combined mask
    mask = torch.Tensor(np.any(detections.mask, axis=0, keepdims=True))
    return mask, annotated_image


def compute_g_accuracy(
    eval_file: List[dict],
    artifact_root: str = "stage1_eval",
    retry_threshold: float = 0.1
) -> float:
    
    with open(eval_file) as file:
        eval_data = json.load(file)

    logger.info(f"{sum(len(v) for v in eval_data.values() if isinstance(v, list))} evaluation items to process.")

    correct, total = 0,0
    for garment_class, node in _iterate_items(eval_data):
        person = node["person"]
        obj = node.get("object", {})

        tryon_img_path = person["input_tryon_img_path"]
        # Get the tryon cond img
        tryon_img_path = person["input_tryon_img_path"]

        prompt = "replacing the dress"

        base_name = os.path.splitext(os.path.basename(tryon_img_path))[0]

        # 1st pass
        try:
            mask, annotated = generate_mask(img_path=tryon_img_path, prompt=prompt, box_threshold=BOX_THRESHOLD)
        except Exception as e:
            logger.error(f"Initial generate_mask failed for {tryon_img_path}: {e}")
            mask = torch.zeros((1, 1, 1), dtype=torch.uint8)  # dummy no-detect
            annotated = None

        if mask.eq(0).all():
            logger.info(f"No object detected, retrying with lower threshold {retry_threshold}")
            try:
                mask, annotated = generate_mask(img_path=tryon_img_path, prompt=prompt, box_threshold=retry_threshold)
            except Exception as e:
                logger.error(f"Retry generate_mask failed for {tryon_img_path}: {e}")
                mask = torch.zeros((1, 1, 1), dtype=torch.uint8)
                annotated = None

                if mask.eq(0).all():
                    print("Still no mask after retry with box_threshold=0.1")
                    tryon_img = Image.open(tryon_img_path).convert("RGB")
                    tryon_img = T.ToTensor(tryon_img)
                    mask = torch.zeros_like(tryon_img)

                    total += 1
                else:
                    correct += 1


                # Log mask to MLFlow
                try:
                    mask_img = to_pil(mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask.clamp(0, 1))
                    mlflow.log_image(
                        mask_img,
                        artifact_file=f"{artifact_root}/{garment_class}/masks/{base_name}_mask.png"
                    )
                except Exception as e:
                    logger.debug(f"MLflow mask log failed for {tryon_img_path}: {e}")

                try:
                    # Log annotated image to MLFlow
                    annot_img = Image.fromarray(annotated)
                    mlflow.log_image(
                        annot_img,
                        artifact_file=f"{artifact_root}/{garment_class}/annotations/{base_name}_annot.jpg"
                    )
                except Exception as e:
                    logger.debug(f"MLflow annotation log failed for {tryon_img_path}: {e}")

        if total == 0:
            logger.warning("No valid items processed; returning G-accuracy = 0.0")
            return 0.0

 
        num_no_mask = total - correct
        g_accuracy = (len(eval_data) - num_no_mask) / float(len(eval_data))
        return  g_accuracy 

                
@torch.no_grad()
def compute_clip_t(eval_file, device):

    with open(eval_file) as file:
        eval_data = json.load(file)
    
    similarities = []

    for garment_class, node in _iterate_items(eval_data):
        person = node["person"]
        img_path = person["input_tryon_img_path"]

        model_description = person["tryon_description"]
        if isinstance(model_description, list) and len(model_description) > 0 and isinstance(model_description[0], str) and model_description[0].strip():
            text = model_description[0].strip()
        else:
            logger.info(f"Falling back to {garment_class} as the model desciption for clip-t")
            # fallback to class name
            text = garment_class

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image for CLIP: {img_path} :: {e}")
            continue

        inputs = clip_processor(
            text=[text], 
            images=[image], 
            return_tensors="pt", 
            padding=True
        ).to(device)
        outputs = clip_model(**inputs)
        # Image-text similarity score
        sim = outputs.logits_per_image.squeeze().item()
        similarities.append(sim)

        if len(similarities) == 0:
            logger.warning("No CLIP similarities computed; returning 0.0")
            return 0.0

        avg_clip_t =  float(sum(similarities) / len(similarities))
        return avg_clip_t


def compute_metrics_stage1(eval_file, artifact_root, retry_threshold):

    g_accuracy = compute_g_accuracy(eval_file, artifact_root, retry_threshold)
    logger.info(f"G-Accuracy: {g_accuracy:.4f}")

    clip_t = compute_clip_t(eval_file)
    logger.info(f"CLIP-T similarity: {clip_t:.4f}")

    return g_accuracy, clip_t

    

