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


def haversine_torch(lat1, lon1, lat2, lon2):
    # all in radians, broadcasts
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
    return R * c

def latlon_to_xyz_torch(lat_deg, lon_deg):
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)

def xyz_to_latlon_torch(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = torch.asin(torch.clamp(z, -1.0, 1.0))
    lon = torch.atan2(y, x)
    return torch.rad2deg(lat), torch.rad2deg(lon)

def predict_gps_retrieval(model, image_tensor, knn, ref_lats, ref_lons, sigma_km=500.0):
    """
    model: your trained MultiHead model
    image_tensor: [3, H, W], preprocessed
    knn: fitted NearestNeighbors on ref_feats
    ref_lats, ref_lons: numpy arrays of training GPS
    sigma_km: bandwidth for KDE (tuneable)
    """

    model.eval()
    with torch.no_grad():
        feat = model.encode_image(image_tensor.unsqueeze(0).to(device))
        feat = feat.cpu().numpy()  # [1, D]

    # 1) find nearest neighbors in feature space
    distances, indices = knn.kneighbors(feat, n_neighbors=100)  # [1, K]
    idx = indices[0]

    neigh_lats = torch.from_numpy(ref_lats[idx])
    neigh_lons = torch.from_numpy(ref_lons[idx])

    # 2) compute pairwise geo distances between neighbors (for KDE center)
    lat_rad = torch.deg2rad(neigh_lats)
    lon_rad = torch.deg2rad(neigh_lons)

    # use a simple kernel: w_i = exp(-(d_i^2)/(2*sigma^2)) where d_i is distance from each neighbor to all others
    # to approximate density, we can just weight all neighbors equally or by a simple heuristic.
    # A good practical simplification: weight by closeness to best neighbor in feature space.

    # for now: weight by distance in feature space (smaller feature distance => larger weight)
    # distances is shape [1, K]
    d_feat = torch.from_numpy(distances[0])  # [K]
    # avoid div by zero
    d_feat = d_feat / (d_feat.std() + 1e-6)
    weights = torch.exp(-0.5 * d_feat**2)    # [K]
    weights = weights / weights.sum()

    # 3) compute weighted mean on the unit sphere
    neigh_xyz = latlon_to_xyz_torch(neigh_lats, neigh_lons)  # [K, 3]
    weighted_xyz = (weights[:, None] * neigh_xyz).sum(dim=0)
    weighted_xyz = weighted_xyz / weighted_xyz.norm()        # normalize

    pred_lat, pred_lon = xyz_to_latlon_torch(weighted_xyz[None, :])
    return float(pred_lat), float(pred_lon)

def plot_predictions(pred_lats, pred_lons, true_lats, true_lons):
    # Convert to GeoDataFrame
    pred_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pred_lons, pred_lats),
        crs="EPSG:4326"
    )
    true_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(true_lons, true_lats),
        crs="EPSG:4326"
    )

    # Load world map
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    # Plot
    ax = world.plot(figsize=(10, 5), edgecolor="black")
    true_points.plot(ax=ax, color="yellow", markersize=5, label="True GPS")
    pred_points.plot(ax=ax, color="red", markersize=5, label="Predicted GPS")

    plt.title("Predicted vs True GPS Locations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/result_{lat:.2f}_{lon:.2f}.png", dpi=300)
    plt.close()

def load_knn(npz_path="all_set_data.npz"):
    if not os.path.exists(npz_path):
        print("Extracting features from training set...")
        csv_path = "filtered_images.csv"  # <-- your filtered CSV
        img_root = "images"  # <-- your folder
        
        # resnet_transform = transform = T.Compose([
        #     T.Resize(256),
        #     T.CenterCrop(224),
        #     T.ToTensor(),
        #     T.Normalize(
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])
                    
        train_dataset = GeoImageDataset(
            csv_path=csv_path,
            img_root=img_root,
            transform=model.preprocess,  # use model's own preprocessing
            filter_missing=True,
        )
                    
        # assume train_dataset returns {"image": ..., "lat": ..., "lon": ...}
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)

        all_feats = []
        all_lats = []
        all_lons = []

        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(device)
                # get backbone features only
                feats = model.encode_image(images)   # [B, D] — the same as you use for heads
                # feats = feats['feats'].cpu().numpy()
                feats = feats.cpu().numpy()

                all_feats.append(feats)
                all_lats.append(batch["lat"].numpy())
                all_lons.append(batch["lon"].numpy())

        ref_feats = np.concatenate(all_feats, axis=0)   # shape [N, D]
        ref_lats  = np.concatenate(all_lats, axis=0)    # shape [N]
        ref_lons  = np.concatenate(all_lons, axis=0)    # shape [N]
        
        np.savez(npz_path, feats=ref_feats, lats=ref_lats, lons=ref_lons)
    else:
        print("Loading cached features...")
        data = np.load(npz_path)
        ref_feats = data["feats"]
        ref_lats = data["lats"]
        ref_lons = data["lons"]
    return ref_feats, ref_lats, ref_lons

if __name__ == "__main__":
    img_path_root = "test_img"
    img_paths = os.listdir(img_path_root)
    os.makedirs("results", exist_ok=True)

    print("Loading model...")
    model = MultiHeadGeoCLIP().to(device)
    ckpt = torch.load("checkpoints/geo_cell_full_set/epoch_007.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ref_feats, ref_lats, ref_lons = load_knn("clip_all_set_007.npz")
    knn = NearestNeighbors(n_neighbors=100, metric="euclidean")
    knn.fit(ref_feats)

    R = 6371.0  # Earth radius km

    for img_path in img_paths:
        img = Image.open(os.path.join(img_path_root, img_path))
        
        lat = 0
        lon = 0
        for marker, data in img.applist:
            if marker == "COM":
                data = data.decode("utf-8")
                if "latitude: " in data:
                    data = data.replace("latitude: ", "")
                    lat = float(data[0:6])
                if "longitude: " in data:
                    data = data.replace("longitude: ", "")
                    lon = float(data[0:6])
        
        # resnet_transform = transform = T.Compose([
        #     T.Resize(256),
        #     T.CenterCrop(224),
        #     T.ToTensor(),
        #     T.Normalize(
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ])

        # pred_lat, pred_lon = predict_gps_retrieval(model, resnet_transform(img), knn, ref_lats, ref_lons, sigma_km=500.0)
        pred_lat, pred_lon = predict_gps_retrieval(model, model.preprocess(img), knn, ref_lats, ref_lons, sigma_km=500.0)
        
        print(f"Predicted GPS: ({pred_lat:.6f}, {pred_lon:.6f})")
        print(f"Ground Truth GPS: ({lat:.6f}, {lon:.6f})")
        if not math.isnan(pred_lat) and not math.isnan(pred_lon):
            plot_predictions([pred_lat], [pred_lon], [lat], [lon])

    
    