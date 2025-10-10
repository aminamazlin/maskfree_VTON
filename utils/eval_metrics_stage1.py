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

def detect(
    image: np.ndarray, 
    classes: List[str],
    box_threshold
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
    detections = detect(image=image, classes=classes, box_threshold=BOX_THRESHOLD)

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

    # TODO: save annotation

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
        print(f"{len(eval_data)} evaluation data samples to process")

    correct, total = 0,0
    for garment_class, items in eval_data.items():  # Upper_body, Lower_body, Outerwear, Dress

        # Loop over all the garment items per class
        for index in tqdm(items, desc=f"Computing G-accuracy for {garment_class}"):
            garment_category = index["garment_category"]
            tryon_img_path = index["tryon_img_path"]
            base_name = os.path.splitext(os.path.basename(tryon_img_path))

            try:
                # Generate mask
                mask, annotated = generate_mask(img_path=tryon_img_path , prompt=garment_category)
                print("No object detected, retrying with lower threshold 0.1")

                # try with threshold 0.1 if no mask is generated
                if mask.eq(0).all():
                    try:
                        mask, annotated = generate_mask(img_path=tryon_img_path, prompt=garment_category, box_threshold=retry_threshold)

                        if mask.eq(0).all():
                            print("Still no mask after retry with box_threshold=0.1")
                            tryon_img = Image.open(tryon_img_path).convert("RGB")
                            tryon_img = T.ToTensor(tryon_img)
                            mask = torch.zeros_like(tryon_img)

                        else:
                            correct += 1
                        total += 1

                        # Log mask to MLFlow
                        mask_img = to_pil(mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask.clamp(0, 1))
                        mlflow.log_image(
                            mask_img,
                            artifact_file=f"{artifact_root}/{garment_class}/masks/{base_name}_mask.png"
                        )

                        # Log annotated image to MLFlow
                        annot_img = Image.fromarray(annotated)
                        mlflow.log_image(
                            annot_img,
                            artifact_file=f"{artifact_root}/{garment_class}/annotations/{base_name}_annot.jpg"
                        )

                    except Exception as e:
                        logger.error(f"Failed to generate mask for: {tryon_img_path}. Error: {e}")

            except Exception as e:
                logger.error(f"Failed to generate mask for: {tryon_img_path}. Error: {e}")
    
    num_no_mask = total - correct
    g_accuracy = (len(eval_data) - num_no_mask) / float(len(eval_data))
    return  g_accuracy 

                

def compute_clip_i():
    pass

def compute_metrics_stage1(eval_file, artifact_root, retry_threshold):

    g_accuracy = compute_g_accuracy(eval_file, artifact_root, retry_threshold)
    logger.info(f"G-Accuracy: {g_accuracy:.4f}")

    

