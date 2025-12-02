import os
import pandas as pd
from PIL import Image, UnidentifiedImageError, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GeoImageDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, transform=None, filter_missing: bool = True):
        """
        Args:
            csv_path: path to CSV with columns: IMG_FILENAME, LAT, LON
            img_root: folder where images are stored
            transform: torchvision (or CLIP) transform to apply to each image
            filter_missing: if True, keep only rows whose image file exists
        """
        self.img_root = img_root
        self.transform = transform

        df = pd.read_csv(csv_path)

        required_cols = {"IMG_FILENAME", "LAT", "LON"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Build full paths
        df["IMG_PATH"] = df["IMG_FILENAME"].apply(lambda x: os.path.join(img_root, x))

        if filter_missing:
            exists_mask = df["IMG_PATH"].apply(os.path.isfile)
            n_before = len(df)
            df = df[exists_mask].reset_index(drop=True)
            n_after = len(df)
            print(f"[GeoImageDataset] Kept {n_after}/{n_before} rows with existing images.")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["IMG_PATH"]
        lat = float(row["LAT"])
        lon = float(row["LON"])
        s3_labels = int(row["S3_Label"])
        s16_labels = int(row["S16_Label"])
        s365_labels = int(row["S365_Label"])
        prob_indoor = float(row["Prob_indoor"])
        prob_natural = float(row["Prob_natural"])
        prob_urban = float(row["Prob_urban"])

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            # truncated / corrupt / unreadable
            print(f"[Warning] Skipping bad image ({e}): {img_path}")
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {
            "image": image,
            "lat": torch.tensor(lat, dtype=torch.float32),
            "lon": torch.tensor(lon, dtype=torch.float32),
            "s3_label": torch.tensor(s3_labels, dtype=torch.long),
            "s16_label": torch.tensor(s16_labels, dtype=torch.long),
            "s365_label": torch.tensor(s365_labels, dtype=torch.long),
            "prob_indoor": torch.tensor(prob_indoor, dtype=torch.float32),
            "prob_natural": torch.tensor(prob_natural, dtype=torch.float32),
            "prob_urban": torch.tensor(prob_urban, dtype=torch.float32),
            "img_filename": row["IMG_FILENAME"],
            "img_path": img_path,
        }
