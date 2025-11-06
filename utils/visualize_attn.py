import os
import math
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import cv2
from tryon.models.attention import CustomAttnProcessor
from utils.modules import (
    BasicTransformerBlockForward,
    Transformer2DModelForward,
    UNet2DConditionModelForward,
)

# Global collector used by hooks: { timestep: { layer_name: {attn_map, height, width) } }
attn_maps = {}

def hook_function(name, detach=True):
    # Function that will run after a layer does a forward pass
    def forward_hook(module, input, output):
        # module: the layer being called
        # input: what was passed into the layer
        # output: what the layer returned
        proc = getattr(module, "processor", None)
        if proc is None:
            return

        attn_map = getattr(proc, "attn_map", None)
        timestep = getattr(proc, "timestep", None)
        height = getattr(proc, "height", None)
        width = getattr(proc, "width", None)

        # If no map, nothing to store
        if attn_map is None:
            return

        # Store and optionally detach to CPU
        # Use 'unknown' if timestep wasn't propagated correctly to avoid dropping maps
        t_key = int(timestep) if isinstance(timestep, int) else (timestep if timestep is not None else "unknown")
        attn_maps[t_key] = attn_maps.get(t_key, dict())
        stored = attn_map.cpu() if detach else attn_map
        attn_maps[t_key][name] = {"map": stored, "height": height, "width": width}

        # Free per-call buffer
        try:
            delattr(proc, "attn_map")
        except Exception:
            pass

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        # If name is not cross attn
        if not name.endswith(target_name):
            continue

        # Ensure the processor is our custom one and enabled to store maps
        try:
            if isinstance(module.processor, CustomAttnProcessor):
                module.processor.store_attn_map = True
            else:
                module.processor = CustomAttnProcessor(store_attention=True)
        except Exception:
            # Fallback: attempt direct replacement even if attribute missing
            try:
                setattr(module, "processor", CustomAttnProcessor(store_attention=True))
            except Exception:
                pass
        
        hook = module.register_forward_hook(hook_function(name))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model

def init_unet(unet):
    unet = register_cross_attention_hook(unet, hook_function, "attn2")
    unet = replace_call_method_for_unet(unet)
    return unet


def process_token(token, startofword):
        if '</w>' in token:
            token = token.replace('</w>', '')
            if startofword:
                token = '<' + token + '>'
            else:
                token = '-' + token + '>'
                startofword = True
        elif token not in ['<|startoftext|>', '<|endoftext|>']:
            if startofword:
                token = '<' + token + '-'
                startofword = False
            else:
                token = '-' + token + '-'
        return token, startofword


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Overlay attention mask on image as heatmap.
       img: HxW x 3 float32 in [0,1]
       mask: H x W float32 in [0,1]
       returns: H x W x 3 uint8 in [0,255]
    """
    image_weight = 0.5

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET,)
    # Convert BGR to RGB 
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Scale heatmap to 0-1
    heatmap = np.float32(heatmap) / 255.0

    # Makes sure the input img is correcly scaled
    if np.max(img) > 1.0:
        raise Exception("The input image should be np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    # Blend heatmap with img
    cam = (1 - image_weight) * heatmap + image_weight * img
    # Normalize
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ---------------------------
# Small reusable utilities
# ---------------------------
def normalize_attention_map(attn_per_token: torch.Tensor) -> torch.Tensor:
    """Normalize a single attention map to [0, 1]."""
    attn_per_token = attn_per_token.to(torch.float32)
    attn_min = attn_per_token.min()
    attn_max = attn_per_token.max()
    return (attn_per_token - attn_min) / (attn_max - attn_min + 1e-6)


def tensor_img_to_np01(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW torch image tensor to HWC numpy in [0,1].
    Assumes image is 3xHxW. If values look like [-1,1], remap to [0,1].
    """
    img = image.clone().to(torch.float32)
    # if pixels in range ~[-1, 1]
    if torch.amin(img) < 0.1:
        img = (img + 1.0) * 0.5
    img = img.clamp(0.0, 1.0)
    img_np = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # (C,H,W)->(H,W,C)
    return img_np


def resize_attn_to_image(attn_np: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize attention (H,W) to match image size (H,W) and clip to [0,1]."""
    th, tw = target_hw
    attn_map_resized = cv2.resize(attn_np.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.clip(attn_map_resized, 0.0, 1.0)


def save_attn_token_image(
    attn_per_token: torch.Tensor,
    image: torch.Tensor | None,
    out_path: str,
    to_pil: ToPILImage,
):
    """Save a single token attention either overlaid on the image or as grayscale."""
    attn_normalized = normalize_attention_map(attn_per_token)

    if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[0] == 3:
        img_np = tensor_img_to_np01(image)
        attn_np = attn_normalized.detach().cpu().numpy()
        attn_resized = resize_attn_to_image(attn_np, img_np.shape[:2])
        result_img = show_cam_on_image(img_np, attn_resized)
        Image.fromarray(result_img).save(out_path)
    else:
        to_pil(attn_normalized.to(torch.float32)).save(out_path)


def save_batch_token_images(
    tokens,
    attn_per_batch: torch.Tensor,
    image: torch.Tensor | None,
    batch_dir: str,
    to_pil: ToPILImage,
):
    """Iterate all tokens for a single batch and save visualizations."""
    startofword = True
    for i, (token, attn_per_token) in enumerate(zip(tokens, attn_per_batch[: len(tokens)])):
        token_clean, startofword = process_token(token, startofword)
        out_path = os.path.join(batch_dir, f"{i}-{token_clean}.png")
        save_attn_token_image(attn_per_token, image, out_path, to_pil)


def reshape_attn_to_spatial(attn_tensor, height=None, width=None):
    """Reshape attention from (B, heads, q_len, k_len) to (B, heads, H, W, k_len).

    If height/width are provided, validate that H*W == q_len. Otherwise, try to
    infer a square grid. Returns None if it cannot reshape reliably.
    """
    b, heads, q_len, k_len = attn_tensor.shape
    if height is None or width is None:
        # Compute spatial attention map
        r = int(math.sqrt(q_len))
        if r * r == q_len:
            h, w = r, r
            # Reshape the 4D tensor -> 5D, turning the flat query dimension into a 2D grid
            attn_spatial = attn_tensor.view(b, heads, h, w, k_len)
        # If original img feat map was not a square grid
        else:
            print("Can't reshape attention tensor to a spatial map")
    if height * width != q_len:
        print("Given height * width is not equal to the query length tokens")

    return attn_spatial


def save_attn_maps(
    attn_maps, 
    tokens, 
    images: torch.Tensor, 
    max_batches: int,
    out_dir='attn_maps', 
    unconditional=True,
    save_per_layer: bool = True):
    """
     - unconditional: if True, assume batch is [uncond, cond] and use conditioned half
    """

    os.makedirs(out_dir, exist_ok=True)
    to_pil = ToPILImage()

    # tokens for each batch element
    all_tokens = tokens

    # Check if its RGB and move to CPU (B x 3 x H x W)
    image_list = None
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[1] == 3:
        image_list = images.detach().cpu().to(torch.float32)

    for timestep, layers in attn_maps.items():
        # Reset per-timestep accumulators
        t_total_map = None  # sum of attention maps from all layers
        t_total_shape = None
        t_count = 0  # number of layers that contributed to t_total_map
        timestep_dir = os.path.join(out_dir, f"{timestep}")
        os.makedirs(timestep_dir, exist_ok=True)

        if save_per_layer:
            per_layer_root = os.path.join(timestep_dir, 'per_layer')
            os.makedirs(per_layer_root, exist_ok=True)
        
        for layer, payload in layers.items():
            if save_per_layer:
                layer_dir = os.path.join(per_layer_root, f"{layer}")
                os.makedirs(layer_dir, exist_ok=True)
            
            # Get the attention map & its weight and height
            if isinstance(payload, dict):
                attn_tensor = payload.get("map")
                h = payload.get("height")
                w = payload.get("width")
            else:
                print("No complete payload found")

            if attn_tensor is None:
                print("No attention maps found")
                continue

            # Expect (B, heads, q_len, k_len)
            if attn_tensor.dim() == 4:
                # gives back (B, heads, h, w, k_len)
                attn_spatial = reshape_attn_to_spatial(attn_tensor, h, w)
                if attn_spatial is None:
                    # Skip this layer if we can't reshape
                    continue

            # Sum over heads
            attn_map = attn_spatial.sum(1)  # (B, heads, h, w, k_len) -> (B, h, w, k_len) 

            # If unconditional pairing used, take conditioned half
            if unconditional and attn_map.shape[0] >= 2:
                # split the batch into two halves, keep the 2nd half (conditioned)
                attn_map = attn_map.chunk(2)[1] 

            # Before: (B, h, w, k_len)
            # After:  (B, k_len, h, w)
            attn_map = attn_map.permute(0, 3, 1, 2) # move token dim to channel

            # Set attn map shape from the first valid layer
            if t_total_map is None:
                # Create empty tensor
                t_total_map = torch.zeros_like(attn_map)
                # Get h, w
                t_total_shape = attn_map.shape[-2:]
            
            
            # For subseuent layers, resize to accumulator shape if needed 
            if attn_map.shape[-2:] != t_total_shape:
                attn_map_resized = F.interpolate(attn_map, size=t_total_shape, mode='bilinear', align_corners=False)
            else:
                attn_map_resized = attn_map
            
            t_total_map = t_total_map + attn_map_resized
            t_count += 1

            # Iterate over all attn maps for each batch
            for batch_idx, (tokens, attn_per_batch) in enumerate(zip(all_tokens, attn_map)):   # attn_map contains for each batch, the attn maps for each token
                if batch_idx >= max_batches:
                    break
                # create dir for batch
                batch_dir = os.path.join(layer_dir, f"batch-{batch_idx}")
                os.makedirs(batch_dir, exist_ok=True)
                
                # Get the img for this batch
                image = None
                if image_list is not None and batch_idx < image_list.shape[0]:
                    image = image_list[batch_idx]
                
                # Save all token maps for this batch
                save_batch_token_images(tokens, attn_per_batch, image, batch_dir, to_pil)



        # After iterating all layers in this timestep, save per-timestep averaged maps under <timestep>/all/
        if t_count > 0:
            # Average attn across all layers per timestep
            avg_attn = t_total_map / t_count  # (B, k_len, H, W)
            model_dir = os.path.join(timestep_dir, 'all')
            os.makedirs(model_dir, exist_ok=True)


            for batch_idx, (attn_map, tokens) in zip(avg_attn, all_tokens):
                if batch_idx >= max_batches:
                    break
                batch_dir = os.path.join(model_dir, f'batch-{batch_idx}')
                os.makedirs(batch_dir, exist_ok=True)

                image = None
                if image_list is not None and batch_idx < image_list.shape[0]:
                    image = image_list[batch_idx]

                save_batch_token_images(tokens, attn_per_batch, image, batch_dir, to_pil)



                            



   