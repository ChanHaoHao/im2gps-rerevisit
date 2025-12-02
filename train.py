from torch.utils.data import DataLoader
from geo_dataset import GeoImageDataset  # your file name here
from model import MultiHeadGeoCLIP, device, ResNet18MultiHead
from loss import multitask_loss
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import torchvision.transforms as T

csv_path = "filtered_images.csv"  # <-- your filtered CSV
img_root = "images"  # <-- your folder
save_dir = "./checkpoints/resnet18_multihead/"
os.makedirs(save_dir, exist_ok=True)

# 1) create model first so we can use its preprocess
# model = MultiHeadGeoCLIP().to(device)
model = ResNet18MultiHead().to(device)
print("Finish loading model.")

# only train the head if CLIP is frozen
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def load_checkpoint(ckpt_path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)

    # restore weights
    model.load_state_dict(ckpt["model_state_dict"])

    # restore optimizer (if you saved it and passed one)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # figure out which epoch to start from next
    start_epoch = ckpt.get("epoch", -1) + 1


    print(f"Loaded checkpoint '{ckpt_path}' (epoch {start_epoch-1})")
    return start_epoch

ckpts = sorted(glob.glob(os.path.join(save_dir, "epoch_*.pth")))
if len(ckpts)>0:
    last_ckpt = ckpts[-1]    # e.g. 'epoch_009.pth'
    start_epoch = load_checkpoint(last_ckpt, model, optimizer, device=device)
else:
    start_epoch = 0

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
train_dataset = GeoImageDataset(
    csv_path=csv_path,
    img_root=img_root,
    transform=resnet_transform,
    filter_missing=True,
)
print(f"Train dataset size: {len(train_dataset)} images.")

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
)
print("Finish creating DataLoader.")

for i in range(4):
    epoch = start_epoch + i
    model.train()

    # running sums for averaged epoch losses
    total_loss = 0.0
    total_loss_s3 = 0.0
    total_loss_s16 = 0.0
    total_loss_s365 = 0.0
    total_samples = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        batch_size = images.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        preds = model(images)

        loss, aux = multitask_loss(preds, batch)
        loss.backward()
        optimizer.step()

        # accumulate weighted losses
        total_loss += loss.item() * batch_size
        total_loss_s3 += aux["loss_s3"].item() * batch_size
        total_loss_s16 += aux["loss_s16"].item() * batch_size
        total_loss_s365 += aux["loss_s365"].item() * batch_size

    # compute average losses for the epoch
    avg_loss = total_loss / total_samples
    avg_s3 = total_loss_s3 / total_samples
    avg_s16 = total_loss_s16 / total_samples
    avg_s365 = total_loss_s365 / total_samples

    # ---- Save checkpoint ----
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)

    print(
        f"Epoch {epoch}: "
        f"loss={avg_loss:.4f}, "
        f"s3={avg_s3:.4f}, "
        f"s16={avg_s16:.4f}, "
        f"s365={avg_s365:.4f}"
    )
