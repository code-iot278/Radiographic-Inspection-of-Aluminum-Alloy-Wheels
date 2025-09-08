import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.filters import frangi
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import regnet_y_400mf, mobilenet_v3_small

# -----------------------------
# Handcrafted Features
# -----------------------------
def extract_handcrafted_features(image, mask):
    feats = {}
    labeled = label(mask)
    props = regionprops(labeled, intensity_image=image)
    if len(props) > 0:
        p = props[0]
        feats["area"] = p.area
        perimeter = p.perimeter if p.perimeter > 0 else 1
        feats["circularity"] = 4*np.pi*p.area/(perimeter**2)
        feats["aspect_ratio"] = p.major_axis_length / (p.minor_axis_length + 1e-5)
        feats["mean_intensity"] = np.mean(image[mask > 0]) if np.any(mask>0) else 0
        feats["median_intensity"] = np.median(image[mask > 0]) if np.any(mask>0) else 0

        mean_fg = np.mean(image[mask > 0]) if np.any(mask > 0) else 0
        mean_bg = np.mean(image[mask == 0]) if np.any(mask == 0) else 0
        feats["local_contrast"] = mean_fg - mean_bg

        edges = cv2.Canny(image, 100, 200)
        feats["edge_density"] = np.sum(edges[mask > 0]) / (p.area + 1e-5)

        frangi_resp = frangi(image.astype(float)/255.0)
        feats["frangi_mean"] = np.mean(frangi_resp[mask > 0]) if np.any(mask>0) else 0
        feats["frangi_var"] = np.var(frangi_resp[mask > 0]) if np.any(mask>0) else 0

    # Texture: LBP
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    feats["lbp_hist_mean"] = np.mean(np.histogram(lbp, bins=np.arange(11))[0])

    # Texture: HOG (mini)
    hog_feat = hog(image, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=False)
    feats["hog_mean"] = np.mean(hog_feat)
    feats["hog_var"] = np.var(hog_feat)

    # Texture: GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0,np.pi/2], symmetric=True, normed=True)
    feats["glcm_contrast"] = np.mean(graycoprops(glcm, 'contrast'))
    feats["glcm_homogeneity"] = np.mean(graycoprops(glcm, 'homogeneity'))
    feats["glcm_energy"] = np.mean(graycoprops(glcm, 'energy'))
    feats["glcm_corr"] = np.mean(graycoprops(glcm, 'correlation'))

    return feats

# -----------------------------
# Tiny Learned Embeddings (MobileViT-S + RegNetY-400MF)
# -----------------------------
class DropoutWrapper(nn.Module):
    def __init__(self, model, feat_dim, p=0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p)
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return self.dropout(x)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3 (proxy for MobileViT-S, lighter)
mobilenet = mobilenet_v3_small(pretrained=True)
mobilenet_headless = nn.Sequential(*list(mobilenet.children())[:-1])
mv_model = DropoutWrapper(mobilenet_headless, 576).to(device).eval()

# RegNetY-400MF
regnet = regnet_y_400mf(pretrained=True)
regnet_headless = nn.Sequential(*list(regnet.children())[:-1])
reg_model = DropoutWrapper(regnet_headless, 400).to(device).eval()

def extract_deep_features(img_path, n_samples=8):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    mv_feats, rg_feats = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            mv = mv_model(x).cpu().numpy().flatten()
            rg = reg_model(x).cpu().numpy().flatten()
            mv_feats.append(mv)
            rg_feats.append(rg)

    mv_feats = np.array(mv_feats)
    rg_feats = np.array(rg_feats)
    out = {
        "mv_mean": np.mean(mv_feats),
        "mv_var": np.var(mv_feats),
        "rg_mean": np.mean(rg_feats),
        "rg_var": np.var(rg_feats)
    }
    return out

# -----------------------------
# Process All Images
# -----------------------------
def process_images(input_folder, output_csv):
    all_features = []
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        hand_feats = extract_handcrafted_features(img, mask)
        deep_feats = extract_deep_features(img_path)

        combined = {"filename": fname}
        combined.update(hand_feats)
        combined.update(deep_feats)
        all_features.append(combined)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")
    return df

# -----------------------------
# Run Example
# -----------------------------
input_folder = "/content/drive/MyDrive/Colab Notebooks/Wheel/output_segmentation_gray"
output_csv = "/content/drive/MyDrive/Colab Notebooks/Wheel/features_lightweight.csv"

df = process_images(input_folder, output_csv)
df.head()
