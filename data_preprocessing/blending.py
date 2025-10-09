import torchvision.transforms as T
import torch, torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision
import re
from torchvision.transforms.functional import to_tensor


def dilate_mask():
    pass

def resize_img_mask(image_paths, mask_paths):
    pass

def sigmoid(x, scale=1.0):
            return 1 / (1 + np.exp(-scale * x))

def blend_orig_garment(
    old_tryon: torch.Tensor, new_tryon: torch.Tensor,
    old_mask: torch.Tensor, new_mask: torch.Tensor,
    dilate_kernel: int = 21,
    dilate_iter: int = 1,
    sigmoid_weight: float = 16.0,   # steeper edge than 12.0
    blur_alpha_kernel: int = 0, blur_alpha_sigma: float = 0.0,
    post_sharpen_strength: float = 0.0
):
    # Resize old image to match new tryon image
    H, W = new_tryon.shape[1:]
    resize_img = lambda x: x if x.shape[1:]==(H,W) else F.interpolate(x.unsqueeze(0), (H,W), mode="bicubic", align_corners=False).squeeze(0)
    resize_mask = lambda x: x if x.shape[1:]==(H,W) else F.interpolate(x.unsqueeze(0),(H,W), mode="nearest").squeeze(0)

    old_tryon = resize_img(old_tryon)
    new_tryon = resize_img(new_tryon)

    # Add the channel dimension: H,W -> C,H,W
    if old_mask.ndim == 2:
        old_mask = old_mask.unsqueeze(0)

    if new_mask.ndim == 2:
        new_mask = new_mask.unsqueeze(0)
    
    # Resize old mask to match new tryon image
    old_mask = resize_mask((old_mask > 0.5).float())
    new_mask = resize_mask((new_mask > 0.5).float())

    # Union the old garment + new garment mask
    union_mask = torch.clamp(old_mask + new_mask, 0, 1)
    # union_np = union_mask[0].cpu().numpy().astype(np.float32)
    union_np = union_mask.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dilated_mask = cv2.dilate((union_np > 0.5).astype(np.uint8), kernel, iterations=dilate_iter).astype(np.float32)

    # Get dilated region
    blend_mask = union_np.copy()
    boundary_region = (dilated_mask - blend_mask) > 0   # Boolean

    if np.any(boundary_region):
        # for every background px, how far is it from the union mask
        # dist_from_garment = distance_transform_edt(1.0 - union_mask)
        dist_from_garment = distance_transform_edt(1.0 - union_np)
        # Max distance
        boundary_width = max(1e-6, np.max(dist_from_garment[boundary_region]))
        normalized_dist = 1.0 - dist_from_garment[boundary_region] / boundary_width
        # Make a smooth feather from the union mask edge
        sigmoid_input = sigmoid_weight * normalized_dist - (sigmoid_weight/2)
        blend_mask[boundary_region] = sigmoid(sigmoid_input)

    blend_mask = torch.from_numpy(blend_mask).to(new_tryon.device, dtype=new_tryon.dtype).unsqueeze(0)  # [1,H,W]

    # optional tiny smoothing of the dilated region (keep small to preserve edge crispness)
    if blur_alpha_kernel and blur_alpha_sigma % 2 == 1 and blur_alpha_sigma > 0:
        gaussian_blur = GaussianBlur(kernel_size=(blur_alpha_kernel, blur_alpha_kernel), sigma=blur_alpha_sigma)
        blend_mask = gaussian_blur(blend_mask)

    # Blending
    final_tryon = new_tryon * (1 - blend_mask) + old_tryon * blend_mask

    # optional sharpening
    if post_sharpen_strength > 0:
        gaussian_blur = GaussianBlur(kernel_size=(5,5), sigma=1.0)
        blurred_tryon =  gaussian_blur(final_tryon)
        final_tryon = torch.clamp(final_tryon + post_sharpen_strength * (final_tryon - blurred_tryon), 0, 1)
    
    return final_tryon.clamp(0,1)


if __name__ == "__main__":
    old_tryon_dir = Path("~/desktop/preprocess_dress/new_cloth").expanduser()
    new_data_dir  = Path("~/desktop/preprocess_dress").expanduser()
    # new_tryon_dir = new_data_dir / "original_cloth"
    new_tryon_dir = new_data_dir / "original_cloth"
    new_mask_dir  = new_data_dir / "original_masks"
    old_mask_dir  = new_data_dir / "new_masks"

    out_dir = Path("~/desktop/preprocess_dress/new_tryon_5").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if directories exist
    for dir_path, name in [(old_tryon_dir, "old_tryon_dir"), 
                           (new_tryon_dir, "new_tryon_dir"),
                           (new_mask_dir, "new_mask_dir"),
                           (old_mask_dir, "old_mask_dir")]:
        if not dir_path.exists():
            print(f"❌ Directory doesn't exist: {dir_path} ({name})")
            raise SystemExit(1)
        else:
            print(f"✅ Found directory: {dir_path}")

    # Collect new try-on images (jpg/png)
    new_tryon_images = sorted(new_tryon_dir.glob("*.jpg")) + sorted(new_tryon_dir.glob("*.png"))
    if not new_tryon_images:
        print(f"⚠️ No images found in {new_tryon_dir}")
    
    # Keep track of missing items
    missing_entries = []   # list of strings (one per missing item)
    missing_counts = {"old_img":0, "new_mask":0, "old_mask":0, "load_error":0}

    for new_img_path in tqdm(new_tryon_images, desc="Generating new try-on images"):
        stem = new_img_path.stem

        # Expected counterparts
        old_img_path  = old_tryon_dir / f"{stem}.png"
        new_mask_path = new_mask_dir  / f"{stem}.png"
        old_mask_path = old_mask_dir  / f"{stem}.png"

        # Check existence
        missing_this = []
        if not old_img_path.exists():
            missing_this.append(("old_img", old_img_path))
        if not new_mask_path.exists():
            missing_this.append(("new_mask", new_mask_path))
        if not old_mask_path.exists():
            missing_this.append(("old_mask", old_mask_path))

        if missing_this:
            tags = ", ".join([t for t, _ in missing_this])
            print(f"[skip {stem}] Missing: {tags}")
            for tag, p in missing_this:
                missing_counts[tag] += 1
                missing_entries.append(f"{stem}\t{tag}\t{p}")
            continue

        # Try to open/convert; catch late errors (e.g., corrupt files)
        try:
            old_img  = to_tensor(Image.open(old_img_path).convert("RGB"))
            new_img  = to_tensor(Image.open(new_img_path).convert("RGB"))
            old_mask = to_tensor(Image.open(old_mask_path).convert("L"))
            new_mask = to_tensor(Image.open(new_mask_path).convert("L"))
        except Exception as e:
            print(f"[skip {stem}] Load error: {e}")
            missing_counts["load_error"] += 1
            missing_entries.append(f"{stem}\tload_error\t{e}")
            continue

        # out = blend_orig_garment(
        #     old_img, new_img, old_mask, new_mask,
        #     dilate_kernel=10,
        #     dilate_iter=5,
        #     sigmoid_weight=8.0,
        #     blur_alpha_kernel=3,
        #     blur_alpha_sigma=2.0,
        #     post_sharpen_strength=0,
        # )

        out = blend_orig_garment(
            old_img, new_img, old_mask, new_mask,
            dilate_kernel=10,
            dilate_iter=3,
            sigmoid_weight=6.0,
            blur_alpha_kernel=5,
            blur_alpha_sigma=1.5,
            post_sharpen_strength=0,
        )

        out_path = out_dir / f"{stem}.jpg"
        torchvision.utils.save_image(out, out_path)

    # Write the report
    report_path = out_dir / "missing_files.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# stem\twhat\tpath_or_error\n")
        for line in missing_entries:
            f.write(line + "\n")

        # summary footer
        f.write("\n# Summary\n")
        for k, v in missing_counts.items():
            f.write(f"# {k}: {v}\n")

    print(f"Done. Saved outputs to: {out_dir}")
    print(f"Missing items written to: {report_path}")
