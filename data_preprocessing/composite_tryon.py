from pathlib import Path
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
from typing import Optional
import time
import signal

p_background = 0.5
p_foreground = 0.7

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred")

def _alpha_mask(img_rgba: Image.Image) -> np.ndarray:
    """Return alpha channel as a boolean mask (True where visible)."""
    return (np.array(img_rgba.split()[-1], dtype=np.uint8) > 0)

def _list_files(folder: str):
    path = Path(folder)
    files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png"))) 
    return files

def _build_model_mask_canvas(model_img: Image.Image, canvas_w: int, canvas_h: int, model_x: int) -> np.ndarray:
    """
    Place the model's local alpha mask into a full-size canvas mask at (model_x, 0).
    """
    mask = _alpha_mask(model_img.convert("RGBA"))
    mh, mw = mask.shape
    canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

    # Clip into canvas bounds
    x0 = max(0, model_x)
    x1 = min(canvas_w, model_x + mw)
    y0 = 0
    y1 = min(canvas_h, mh)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = mask[y0 - 0:y1 - 0, x0 - model_x:x1 - model_x]
    return canvas

def _sample_xy_overlap_both(
    mask1: np.ndarray,
    mask2: np.ndarray,
    fg_mask: np.ndarray,
    min_overlap_frac: float,
    max_overlap_frac: float,
    canvas_w: int,
    canvas_h: int,
):
    """
    Find one (x,y) where foreground overlaps BOTH masks within the given fractions.
    Overlap is measured as a fraction of foreground visible pixels.
    Will timeout after 60 seconds.
    """
    fh, fw = fg_mask.shape
    fg_area = int(fg_mask.sum())
    if fg_area == 0:
        return None

    min_px = max(1, int(fg_area * min_overlap_frac))
    max_px = int(fg_area * max_overlap_frac)

    min_x, max_x = 0, max(0, canvas_w - fw)
    min_y, max_y = 0, max(0, canvas_h - fh)

    # Check if foreground can fit in canvas
    if min_x > max_x or min_y > max_y:
        return None

    # Set up timeout alarm for 60 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 seconds timeout
    
    try:
        while True:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)

            # Ensure we don't go out of bounds
            reg1_h = min(fh, canvas_h - y)
            reg1_w = min(fw, canvas_w - x)
            
            if reg1_h <= 0 or reg1_w <= 0:
                continue
                
            reg1 = mask1[y:y+reg1_h, x:x+reg1_w]
            reg2 = mask2[y:y+reg1_h, x:x+reg1_w]
            fg_region = fg_mask[:reg1_h, :reg1_w]

            ov1 = int((fg_region & reg1).sum())
            ov2 = int((fg_region & reg2).sum())

            if (min_px <= ov1 <= max_px) and (min_px <= ov2 <= max_px):
                signal.alarm(0)  # Cancel the alarm
                return x, y
    except TimeoutException:
        signal.alarm(0)  # Cancel the alarm
        raise TimeoutException("Could not find valid position within 60 seconds")

def composite_background(background_img: Image.Image, model_img: Image.Image, x: int = None) -> Image.Image:
    if not background_img:
        raise SystemExit("No valid background found")

    if not model_img:
        raise SystemExit("No valid model image found")
    
    # Get bg size
    W, H = background_img.size
    
    # Create a copy of background to avoid modifying original
    composite = background_img.copy()

    composite.alpha_composite(model_img, dest=(x, 0))
        
    return composite

def composite_images(
    new_tryon_img: Image.Image,
    orig_tryon_img: Image.Image,
    foreground_img: Image.Image,
    background_img: Image.Image = None,
    new_model_rgba: Optional[Image.Image] = None,
    orig_model_rgba: Optional[Image.Image] = None,
    model_x: int = 0,                      # <-- NEW: shared X comes from caller
    min_overlap: float = 0.02,
    max_overlap: float = 0.30,
    fg_scale: float | None = None,
    fg_angle: float | None = None,
) -> tuple[Image.Image, Image.Image, tuple[int, int], float, float]:
    
    if background_img: 
        # 1) paste both models at the SAME X
        new_comp  = composite_background(background_img, new_tryon_img,  x=model_x)
        orig_comp = composite_background(background_img, orig_tryon_img, x=model_x)

        cw, ch = new_comp.size

        # 2) build canvas masks aligned at the SAME X
        new_mask_canvas  = _build_model_mask_canvas(new_tryon_img,  cw, ch, model_x)
        orig_mask_canvas = _build_model_mask_canvas(orig_tryon_img, cw, ch, model_x)
    elif not background_img and new_model_rgba and orig_model_rgba:
        new_comp = new_tryon_img
        orig_comp = orig_tryon_img

        cw, ch = new_comp.size

        # 2) build canvas masks aligned at the SAME X (use model_x = 0 for no background)
        new_mask_canvas  = _build_model_mask_canvas(new_model_rgba,  cw, ch, 0)
        orig_mask_canvas = _build_model_mask_canvas(orig_model_rgba, cw, ch, 0)
    else:
        raise SystemExit("Invalid configuration: missing required parameters")
        
        
    # 3) shared FG transform (scale/angle), then mask AFTER transform
    fg = foreground_img.convert("RGBA")
    scale = fg_scale if fg_scale is not None else random.uniform(0.25, 0.6)
    angle = fg_angle if fg_angle is not None else random.uniform(-45, 45)
    fg = fg.resize((max(1, int(fg.width*scale)), max(1, int(fg.height*scale))), Image.LANCZOS)
    fg = fg.rotate(angle, expand=True, resample=Image.BICUBIC)

    fg_mask = _alpha_mask(fg)
    fh, fw = fg_mask.shape
    if fg_mask.sum() == 0:
        raise SystemExit("Foreground became empty after transform")

    # 4) one (x,y) that satisfies overlap for BOTH masks
    xy = _sample_xy_overlap_both(
        new_mask_canvas, orig_mask_canvas, fg_mask,
        min_overlap, max_overlap, cw, ch
    )
    if xy is None:
        raise SystemExit("Could not find a shared (x,y) that overlaps both models within constraints")
    fg_x, fg_y = xy


    # 5) paste FG at SAME (x,y) on both
    new_out  = new_comp.copy()
    orig_out = orig_comp.copy()
    if not background_img:
        new_out.paste(fg, (fg_x, fg_y), fg)
        orig_out.paste(fg, (fg_x, fg_y), fg)
        return new_out, orig_out, (fg_x, fg_y), scale, angle
    else:
        new_out.alpha_composite(fg,  dest=(fg_x, fg_y))
        orig_out.alpha_composite(fg, dest=(fg_x, fg_y))
        return new_out, orig_out, (fg_x, fg_y), scale, angle


def main(background_dir, orig_tryon_dir, new_tryon_dir, foreground_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timeout folder for problematic images
    timeout_dir = output_dir / "timeout_images"
    timeout_dir.mkdir(parents=True, exist_ok=True)
    
    # List backgrounds & model images
    background_imgs = _list_files(background_dir)
    new_tryon_imgs = _list_files(new_tryon_dir)
    foreground_imgs = _list_files(foreground_dir)

    timeout_images = []
    for new_tryon in tqdm(new_tryon_imgs, desc="Augmenting image data"):

        # Get corresponding original tryon
        name = new_tryon.name
        orig_tryon = orig_tryon_dir / name

        # Choose scale and rotation for foreground
        scale = random.uniform(0.25, 0.5)
        angle = random.uniform(-45, 45)

        # choose random prob for bg
        use_bg = random.random() < p_background
        if use_bg:
            # choose random bg
            background = random.choice(background_imgs)

            # Open in RGBA  
            background_img = Image.open(background).convert("RGBA")
            new_tryon_img = Image.open(new_tryon).convert("RGBA")
            orig_tryon_img = Image.open(orig_tryon).convert("RGBA")
        else:
            # Open in RGB
            new_tryon_rgba = Image.open(new_tryon).convert("RGBA")
            new_tryon_img = Image.open(new_tryon).convert("RGB")
            orig_tryon_rgba = Image.open(orig_tryon).convert("RGBA")
            orig_tryon_img = Image.open(orig_tryon).convert("RGB")


        # choose random prob for fg
        use_fg = random.random() <= p_foreground
        if use_fg:
            # choose random fg
            foreground = random.choice(foreground_imgs)

            # Load images in RGBA
            foreground_img = Image.open(foreground).convert("RGBA")

            if use_bg:
                # Compute a shared X that fits BOTH models on the background
                W, _  = background_img.size
                nw, _ = new_tryon_img.size
                ow, _ = orig_tryon_img.size
                max_x_new  = max(0, W - nw)
                max_x_orig = max(0, W - ow)
                max_x_both = min(max_x_new, max_x_orig)  # must fit both

                shared_x = random.randint(0, max_x_both) if max_x_both > 0 else 0
            else:
                # No background, use image size for positioning
                W, _ = new_tryon_img.size
                shared_x = 0


            # Composite images
            try:
                if use_bg:
                    new_out, orig_out, (fg_x, fg_y), used_scale, used_angle = composite_images(
                        new_tryon_img=new_tryon_img,
                        orig_tryon_img=orig_tryon_img,
                        foreground_img=foreground_img,
                        background_img=background_img,
                        model_x=shared_x,
                        min_overlap=0.1,
                        max_overlap=0.40,
                        fg_scale=scale,   # or None to auto-sample
                        fg_angle=angle,   # or None to auto-sample
                    )
                else:
                    new_out, orig_out, (fg_x, fg_y), used_scale, used_angle = composite_images(
                        new_tryon_img=new_tryon_img,
                        orig_tryon_img=orig_tryon_img,
                        foreground_img=foreground_img,
                        background_img=None,
                        new_model_rgba=new_tryon_rgba,
                        orig_model_rgba=orig_tryon_rgba,
                        model_x=0,  # No background, so use 0
                        min_overlap=0.05,
                        max_overlap=0.40,
                        fg_scale=scale,   # or None to auto-sample
                        fg_angle=angle,   # or None to auto-sample
                    )

            except TimeoutException:
                print(f"Timeout on {name}: Moving to timeout folder")
                # Copy original images to timeout folder
                timeout_new_dir = timeout_dir / "new_tryon" 
                timeout_orig_dir = timeout_dir / "orig_tryon"
                timeout_new_dir.mkdir(parents=True, exist_ok=True)
                timeout_orig_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images to timeout folder
                import shutil
                shutil.copy2(new_tryon, timeout_new_dir / name)
                shutil.copy2(orig_tryon, timeout_orig_dir / name)
                
                timeout_images.append(str(new_tryon))
                continue
            except SystemExit as e:
                print(f"Skipping {name}: {e}")
                continue

            # Create output directories and save images
            new_output_dir = output_dir / "new_tryon"
            orig_output_dir = output_dir / "orig_tryon"
            
            new_output_dir.mkdir(parents=True, exist_ok=True)
            orig_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the composite images
            new_output_path = new_output_dir / name
            orig_output_path = orig_output_dir / name
            
            new_out.save(new_output_path)
            orig_out.save(orig_output_path)
        else:
            # No foreground chosen, just save original images with original background
            new_tryon_img = Image.open(new_tryon).convert("RGB")
            orig_tryon_img = Image.open(orig_tryon).convert("RGB")
   
            
            # Create output directories and save images
            new_output_dir = output_dir / "new_tryon"
            orig_output_dir = output_dir / "orig_tryon"
            
            new_output_dir.mkdir(parents=True, exist_ok=True)
            orig_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the composite images
            new_output_path = new_output_dir / name
            orig_output_path = orig_output_dir / name
            
            new_tryon_img.save(new_output_path)
            orig_tryon_img.save(orig_output_path)
    
    # Save lists of problematic images
    if timeout_images:
        with open(output_dir / "timeout_images.txt", "w") as file:
            for img in timeout_images:
                file.write(f"{img}\n")
    

if __name__ == "__main__":
    background_dir = Path("~/desktop/preprocess_dress/background_images").expanduser()
    foreground_dir = Path("~/desktop/preprocess_dress/foreground_images").expanduser()
    orig_tryon_dir =  Path("~/desktop/preprocess_dress/removed_bg/orig_tryon").expanduser()
    new_tryon_dir =  Path("~/desktop/preprocess_dress/removed_bg/new_tryon").expanduser()
    output_dir = Path("~/desktop/preprocess_dress/augmented_images").expanduser()

    main(
        background_dir=background_dir,
        foreground_dir=foreground_dir,
        orig_tryon_dir=orig_tryon_dir,
        new_tryon_dir=new_tryon_dir,
        output_dir=output_dir
    )



