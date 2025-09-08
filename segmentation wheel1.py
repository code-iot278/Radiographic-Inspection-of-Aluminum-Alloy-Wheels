import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters, color
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import torch
import torchvision.transforms as T
import os
import math
import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Optional: try to import timm to load a TinySwin-like model
try:
    import timm
except Exception:
    timm = None


# ----------------------------
# 1) Porosity map functions
# ----------------------------
def top_hat_porosity(gray, selem_radius=15):
    """
    Compute white and black top-hat transforms and combine to get porosity-like signal.
    gray: grayscale uint8 image
    returns: combined_top_hat normalized float32 [0..1]
    """
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = (255 * (gray - gray.min()) / (gray.max() - gray.min())).astype(np.uint8)

    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*selem_radius+1, 2*selem_radius+1))

    # White top-hat (bright small objects on dark bg)
    white_th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, selem)

    # Black top-hat (dark small objects on bright bg) via closing then subtract
    black_th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, selem)

    combined = cv2.addWeighted(white_th.astype(np.float32), 0.6,
                               black_th.astype(np.float32), 0.4, 0.0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
    return combined.astype(np.float32)


def difference_of_gaussian(gray, sigma1=1.0, sigma2=3.0):
    """
    DoG: blur at two scales and subtract to highlight blob-like structures.
    returns float32 normalized [0..1]
    """
    if gray.dtype != np.uint8:
        gray_u8 = (255 * (gray - gray.min()) / (gray.max() - gray.min() )).astype(np.uint8)
    else:
        gray_u8 = gray

    g1 = cv2.GaussianBlur(gray_u8, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray_u8, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2).astype(np.float32)
    dog = np.abs(dog)
    dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-9)
    return dog.astype(np.float32)


# ----------------------------
# 2) SLIC-Lite (few segments) and superpixel stats
# ----------------------------
def compute_superpixels(image_rgb, n_segments=200, compactness=10.0):
    """
    image_rgb: float image [0..1] or uint8
    returns labels (H,W), and an image with boundaries marked
    """
    img_float = img_as_float(image_rgb)
    labels = slic(img_float, n_segments=n_segments, compactness=compactness, start_label=0)
    boundary_vis = mark_boundaries(img_float, labels)
    return labels, (boundary_vis * 255).astype(np.uint8)


def superpixel_stats_per_band(image, labels):
    """
    image: grayscale or multi-channel (H,W) or (H,W,C)
    labels: superpixel labels
    returns per-pixel map of mean and stddev features aggregated per superpixel
    """
    H, W = labels.shape
    if image.ndim == 2:
        image_stack = image[..., None]
    else:
        image_stack = image  # H,W,C

    C = image_stack.shape[2]
    out_mean = np.zeros((H, W, C), dtype=np.float32)
    out_std = np.zeros((H, W, C), dtype=np.float32)

    # For each label, compute stats
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = (labels == lab)
        if mask.sum() == 0:
            continue
        for c in range(C):
            band = image_stack[..., c]
            vals = band[mask]
            mean = float(vals.mean()) if vals.size else 0.0
            std = float(vals.std()) if vals.size else 0.0
            out_mean[..., c][mask] = mean
            out_std[..., c][mask] = std

    # Optionally normalize per-channel
    for c in range(C):
        m = out_mean[..., c]
        s = out_std[..., c]
        if m.max() > m.min():
            out_mean[..., c] = (m - m.min()) / (m.max() - m.min())
        if s.max() > s.min():
            out_std[..., c] = (s - s.min()) / (s.max() - s.min())

    # Combine mean and std into a feature map (concatenate channels)
    feat = np.concatenate([out_mean, out_std], axis=2)  # H,W,2C
    return feat.astype(np.float32)


# ----------------------------
# 3) Model hook: TinySwin++ placeholder (PyTorch)
# ----------------------------
def load_tinyswin_plus_plus(model_name_or_checkpoint=None, device='cpu'):
    """
    Attempt to load a TinySwin++ model via timm or from local checkpoint path.
    Returns model (eval mode) or None if not available.
    """
    device = torch.device(device)
    model = None
    if timm and model_name_or_checkpoint:
        try:
            # Example: if there's a timm model name that matches TinySwin, use it
            model = timm.create_model(model_name_or_checkpoint, pretrained=True, num_classes=0)  # feature extractor
            model.to(device)
            model.eval()
            print(f"Loaded model from timm: {model_name_or_checkpoint}")
            return model
        except Exception as e:
            print("timm model load failed:", e)

    if model_name_or_checkpoint and os.path.exists(model_name_or_checkpoint):
        # User-specified checkpoint -> user must provide loading code
        ckpt = torch.load(model_name_or_checkpoint, map_location=device)
        # Example: user should adapt below to their saved state_dict/model
        print("Checkpoint loaded; integrate with your model class here.")
        return None

    print("TinySwin++ model not loaded (no timm/model checkpoint). Pipeline will still compute features.")
    return None


# ----------------------------
# 4) Full processing function
# ----------------------------
def process_image_for_porosity(img_bgr,
                               selem_radius=15,
                               dog_sigmas=(1.0, 3.0),
                               n_slic_segments=200,
                               slic_compactness=10.0):
    """
    img_bgr: input BGR image (uint8)
    returns dict of outputs: porosity_map, dog_map, superpixel_labels, superpixel_features, visualization
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Porosity via top-hat
    porosity = top_hat_porosity(gray, selem_radius=selem_radius)

    # DoG
    dog = difference_of_gaussian(gray, sigma1=dog_sigmas[0], sigma2=dog_sigmas[1])

    # Combine porosity + DoG (weighted)
    combined = 0.6 * porosity + 0.4 * dog
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)

    # Superpixels on color image (convert to RGB float)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_float = img_as_float(img_rgb)
    labels, boundary_vis = compute_superpixels(img_rgb_float, n_segments=n_slic_segments, compactness=slic_compactness)

    # Build feature maps: use combined (porosity+dog) and grayscale for stats
    stacked = np.dstack([combined, gray.astype(np.float32) / 255.0])  # H,W,2
    sp_feats = superpixel_stats_per_band(stacked, labels)  # H,W,4 (mean+std per band)

    # Also compute per-superpixel aggregated scalar porosity values (mean)
    unique_labels = np.unique(labels)
    porosity_sp = np.zeros(labels.shape, dtype=np.float32)
    for lab in unique_labels:
        mask = (labels == lab)
        val = combined[mask].mean() if mask.sum() else 0.0
        porosity_sp[mask] = val

    # Final feature tensor for a model: stack the original RGB (normalized), combined porosity, and sp_feats (channel-reduced)
    rgb_norm = img_rgb_float  # already 0..1
    # Reduce sp_feats to two channels (mean of means, mean of stds) to keep size small
    sp_mean = np.mean(sp_feats[..., :sp_feats.shape[2]//2], axis=2, keepdims=True)  # H,W,1
    sp_std = np.mean(sp_feats[..., sp_feats.shape[2]//2:], axis=2, keepdims=True)    # H,W,1

    model_input = np.concatenate([rgb_norm, combined[..., None], sp_mean, sp_std], axis=2)  # H,W,5
    # Convert to CHW tensor for PyTorch
    model_input_tensor = torch.from_numpy(model_input.transpose(2, 0, 1)).float().unsqueeze(0)  # 1,C,H,W

    out = {
        "porosity_map": combined,
        "dog_map": dog,
        "labels": labels,
        "boundary_vis": boundary_vis,
        "superpixel_features": sp_feats,
        "porosity_superpixel_mean": porosity_sp,
        "model_input_tensor": model_input_tensor
    }
    return out


# ----------------------------
# 5) Folder-wise processing + save outputs
# ----------------------------
def process_folder(input_main_folder, output_main_folder,
                   selem_radius=15, dog_sigmas=(1.0, 3.0),
                   n_slic_segments=200, slic_compactness=10.0,
                   model=None, device='cpu'):
    os.makedirs(output_main_folder, exist_ok=True)
    for root, _, files in os.walk(input_main_folder):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                continue
            in_path = os.path.join(root, filename)
            img = cv2.imread(in_path)
            if img is None:
                print("Skipping unreadable:", in_path)
                continue

            results = process_image_for_porosity(img,
                                                selem_radius=selem_radius,
                                                dog_sigmas=dog_sigmas,
                                                n_slic_segments=n_slic_segments,
                                                slic_compactness=slic_compactness)

            # Save outputs in mirrored folder structure
            rel = os.path.relpath(root, input_main_folder)
            save_dir = os.path.join(output_main_folder, rel)
            os.makedirs(save_dir, exist_ok=True)

            base, _ = os.path.splitext(filename)
            # Save porosity map and DoG as images
            por_map = (255 * results["porosity_map"]).astype(np.uint8)
            dog_map = (255 * results["dog_map"]).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"{base}_porosity.png"), por_map)
            cv2.imwrite(os.path.join(save_dir, f"{base}_dog.png"), dog_map)
            # Save boundary visualization
            cv2.imwrite(os.path.join(save_dir, f"{base}_sp_boundaries.png"), cv2.cvtColor(results["boundary_vis"], cv2.COLOR_RGB2BGR))
            # Save labels as a colored visualization (random color per label)
            label_vis = color.label2rgb(results["labels"], bg_label=-1)
            label_vis_u8 = (255 * label_vis).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"{base}_sp_labels.png"), cv2.cvtColor(label_vis_u8, cv2.COLOR_RGB2BGR))

            print("Processed:", in_path)

            # Optional: model inference
            if model is not None:
                model.to(device)
                model.eval()
                with torch.no_grad():
                    inp = results["model_input_tensor"].to(device)
                    feats = model.forward(inp)  # or model(inp) depending on model
                    # Save or process feats as needed (user-specific)
                    print("  Model features shape:", getattr(feats, "shape", "unknown"))

    print("All done.")

# -------------------------
# Input and Output Folders
# -------------------------
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/guided_output/wheel detection image"   # main folder with images + subfolders
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/output_segmentation_gray"  # output folder
os.makedirs(output_main_folder, exist_ok=True)

# -------------------------
# Process All Images (Recursive)
# -------------------------
for root, _, files in os.walk(input_main_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(root, filename)

            # Preserve subfolder structure
            rel_path = os.path.relpath(root, input_main_folder)
            save_dir = os.path.join(output_main_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(save_dir, filename)

            # Load Image in Grayscale
            gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"❌ Skipping {filename}, unable to load.")
                continue

            # Threshold to segment bright/white regions
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Extract White Region (same size, grayscale)
            white_region = cv2.bitwise_and(gray, gray, mask=mask)

            # Save Result
            cv2.imwrite(output_path, white_region)
            print(f"✅ Saved: {output_path}")

            # -------------------------
            # Show Original, Mask, Segmented
            # -------------------------
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1); plt.title("Original Grayscale"); plt.imshow(gray, cmap="gray"); plt.axis("off")
            plt.subplot(1, 3, 2); plt.title("Mask"); plt.imshow(mask, cmap="gray"); plt.axis("off")
            plt.subplot(1, 3, 3); plt.title("Segmented"); plt.imshow(white_region, cmap="gray"); plt.axis("off")
            plt.show()

print("🎉 Processing complete. All segmented grayscale images saved!")