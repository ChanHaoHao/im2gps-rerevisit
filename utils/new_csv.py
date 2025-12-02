import pandas as pd
import os

# Load your CSV
df = pd.read_csv("MP16-Pro/metadata/MP16_Pro_places365.csv")

# Convert IMG_ID to actual filenames
df["IMG_FILENAME"] = df["IMG_ID"].str.replace("/", "_")

# Folder where your images are stored
img_folder = "images"
existing_files = set(os.listdir(img_folder))

print("Start filtering images...")
# Keep only images that exist
df_valid = df[df["IMG_FILENAME"].isin(existing_files)]
print("Filtering complete.")

# Save result
df_valid.to_csv("filtered_images.csv", index=False)

print("Saved filtered_images.csv with", len(df_valid), "rows.")
