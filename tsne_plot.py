import numpy as np
from torch.utils.data import DataLoader
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

csv_path = "filtered_images.csv"  # <-- your filtered CSV
img_root = "images"  # <-- your folder

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
train_clip_dataset = GeoImageDataset(
    csv_path=csv_path,
    img_root=img_root,
    transform=clip_model.preprocess,
    filter_missing=True,
)
print(f"Train dataset size: {len(train_resnet_dataset)} images.")
print("Finish creating DataLoader.")

def extract_feats_and_scene_labels(model, dataset, batch_size=512, device="cuda"):
    """
    dataset: GeoImageDataset that returns keys:
        "image", "prob_indoor", "prob_natural", "prob_urban"
    model: MultiHeadGeoCLIP or ResNet18MultiHead with .encode_image(x) -> [B, D]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    all_feats = []
    all_scene = []  # 0=indoor, 1=natural, 2=urban

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)

            # get features from backbone
            feats = model.encode_image(imgs)            # [B, D]
            feats = feats.cpu().numpy()

            # scene probabilities (assume they come as tensors or numpy arrays)
            p_in  = batch["prob_indoor"].numpy()   # [B]
            p_nat = batch["prob_natural"].numpy()  # [B]
            p_urb = batch["prob_urban"].numpy()    # [B]
            probs = np.stack([p_in, p_nat, p_urb], axis=1)  # [B, 3]

            scene_cls = probs.argmax(axis=1)  # [B], in {0,1,2}

            # optional: filter out "uncertain" samples
            max_prob = probs.max(axis=1)
            mask = max_prob >= 0.5  # or some threshold you like

            feats = feats[mask]
            scene_cls = scene_cls[mask]

            all_feats.append(feats)
            all_scene.append(scene_cls)

    all_feats = np.concatenate(all_feats, axis=0)
    all_scene = np.concatenate(all_scene, axis=0)

    return all_feats, all_scene

resnet_feats, resnet_scene = extract_feats_and_scene_labels(
    resnet_model, train_resnet_dataset, batch_size=512, device=device
)
print("Extracted ResNet18 features:", resnet_feats.shape)

clip_feats, clip_scene = extract_feats_and_scene_labels(
    clip_model, train_clip_dataset, batch_size=512, device=device
)
print("Extracted CLIP features:", clip_feats.shape)

def run_tsne(feats, n_samples=4000, random_state=0):
    if feats.shape[0] > n_samples:
        idx = np.random.choice(feats.shape[0], n_samples, replace=False)
        feats = feats[idx]
    else:
        idx = np.arange(feats.shape[0])

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=random_state,
    )
    emb = tsne.fit_transform(feats)
    return emb, idx

def plot_scene_tsne(emb, scene_cls, title, save_path=None):
    # 0=indoor, 1=natural, 2=urban
    cmap = {0: "tab:blue", 1: "tab:green", 2: "tab:orange"}
    labels = ["indoor", "natural", "urban"]

    plt.figure(figsize=(7, 7))
    for c in [0, 1, 2]:
        mask = (scene_cls == c)
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=6,
            alpha=0.6,
            label=labels[c],
        )
    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        
clip_emb, clip_idx = run_tsne(clip_feats)
plot_scene_tsne(
    clip_emb,
    clip_scene[clip_idx],
    title="CLIP features clustered by indoor/natural/urban",
    save_path="results/clip_tsne_scene.png",
)
print("Saved CLIP t-SNE plot.")

res_emb, res_idx = run_tsne(resnet_feats)
plot_scene_tsne(
    res_emb,
    resnet_scene[res_idx],
    title="ResNet18 features clustered by indoor/natural/urban",
    save_path="results/resnet_tsne_scene.png",
)
print("Saved ResNet18 t-SNE plot.")