import numpy as np
from torch.utils.data import Subset, DataLoader
from geo_dataset import GeoImageDataset
from sklearn.neighbors import NearestNeighbors
from model import MultiHeadGeoCLIP, ResNet18MultiHead, device
import torch
from PIL import Image
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import torchvision.transforms as T
from sklearn.manifold import TSNE
import pandas as pd

csv_path = "filtered_images.csv"  # <-- your filtered CSV
img_root = "images"  # <-- your folder

# 1) read CSV to get urban indices
df = pd.read_csv(csv_path)

# Prob columns from your CSV
p_in  = df["Prob_indoor"].to_numpy()
p_nat = df["Prob_natural"].to_numpy()
p_urb = df["Prob_urban"].to_numpy()

probs = np.stack([p_in, p_nat, p_urb], axis=1)  # [N, 3]

scene_cls = probs.argmax(axis=1)      # 0=indoor,1=natural,2=urban
max_prob  = probs.max(axis=1)

prob_thr = 0.5  # or whatever you like
urban_mask = (scene_cls == 2) & (max_prob >= prob_thr)
urban_idx = np.where(urban_mask)[0]   # indices into df

print("Num urban images:", len(urban_idx))

# lat/lon for urban subset
urban_lats = df["LAT"].to_numpy()[urban_idx]
urban_lons = df["LON"].to_numpy()[urban_idx]

def latlon_to_region_ids(lats, lons, lat_bin_size=20, lon_bin_size=40):
    lat_bins = np.floor((lats + 90)  / lat_bin_size).astype(int)
    lon_bins = np.floor((lons + 180) / lon_bin_size).astype(int)
    lat_bins = np.clip(lat_bins, 0, int(180/lat_bin_size))
    lon_bins = np.clip(lon_bins, 0, int(360/lon_bin_size))
    n_lon_bins = int(360 / lon_bin_size) + 1
    region_ids = lat_bins * n_lon_bins + lon_bins
    return region_ids

region_ids = latlon_to_region_ids(urban_lats, urban_lons)
# ==================================================

clip_model = MultiHeadGeoCLIP().to(device)
clip_ckpt = torch.load("checkpoints/geo_cell_full_set/epoch_007.pth", map_location=device)
clip_model.load_state_dict(clip_ckpt["model_state_dict"])
print("Finish loading CLIP model.")
resnet_model = ResNet18MultiHead().to(device)
resnet_ckpt = torch.load("checkpoints/resnet18_multihead/epoch_011.pth", map_location=device)
resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
print("Finish loading ResNet18 model.")

resnet_transform = transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 2) dataset uses model.preprocess so CLIP gets what it expects
train_resnet_dataset = GeoImageDataset(
    csv_path=csv_path,
    img_root=img_root,
    transform=resnet_transform,
    filter_missing=True,
)
res_urban_dataset = Subset(train_resnet_dataset, urban_idx)
train_clip_dataset = GeoImageDataset(
    csv_path=csv_path,
    img_root=img_root,
    transform=clip_model.preprocess,
    filter_missing=True,
)
clip_urban_dataset = Subset(train_clip_dataset, urban_idx)
print(f"Train dataset size: {len(clip_urban_dataset)} images.")
print("Finish creating DataLoader.")

def extract_feats_only(model, dataset, batch_size=256, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=8)

    all_feats = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            feats = model.encode_image(imgs)      # [B, D]
            feats = feats.cpu().numpy()
            all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats

clip_feats_urban = extract_feats_only(
    clip_model, clip_urban_dataset, batch_size=512, device=device
)
print("Extracted CLIP urban features:", clip_feats_urban.shape)

res_feats_urban = extract_feats_only(
    resnet_model, res_urban_dataset, batch_size=512, device=device
)
print("Extracted ResNet18 urban features:", res_feats_urban.shape)

print(clip_feats_urban.shape[0] == res_feats_urban.shape[0] == region_ids.shape[0])

def run_tsne(clip_feats, resnet_feats, n_samples=4000, random_state=0):
    N = clip_feats.shape[0]
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        clip_feats = clip_feats[idx]
        resnet_feats = resnet_feats[idx]
    else:
        idx = np.arange(N)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(clip_feats)//10)),
        learning_rate=200,
        init="pca",
        random_state=random_state,
    )
    clib_emb = tsne.fit_transform(clip_feats)
    res_emb  = tsne.fit_transform(resnet_feats)
    return clib_emb, res_emb, idx

clip_emb_urban, res_emb_urban, idx = run_tsne(clip_feats_urban, res_feats_urban)

def plot_urban_tsne(emb, region_ids, title, save_path):
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        s=8,
        c=region_ids,
        cmap="tab20",
        alpha=0.7,
    )
    plt.title(title)
    plt.axis("off")
    # plt.colorbar(sc, label="coarse geo region id")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
# CLIP
plot_urban_tsne(
    clip_emb_urban,
    region_ids[idx],
    "CLIP – URBAN only, colored by geo region",
    "results/clip_tsne_urban_regions_new.png",
)
print("Saved CLIP urban t-SNE plot.")

# ResNet18
plot_urban_tsne(
    res_emb_urban,
    region_ids[idx],
    "ResNet18 – URBAN only, colored by geo region",
    "results/resnet_tsne_urban_regions_new.png",
)
print("Saved ResNet18 urban t-SNE plot.")