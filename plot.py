import argparse
import math
import re

import matplotlib.pyplot as plt
import numpy as np

import geopandas as gpd
from shapely.geometry import Point


def parse_log_file(path):
    """
    Parse a .log file for lines like:

        Predicted GPS: (lat, lon)
        Ground Truth GPS: (lat, lon)

    Returns:
        pred (N, 2) array of [lat, lon]
        gt   (N, 2) array of [lat, lon]
    """
    pred_pattern = re.compile(
        r"Predicted GPS:\s*\(([-\d\.]+)\s*,\s*([-\d\.]+)\)"
    )
    gt_pattern = re.compile(
        r"Ground Truth GPS:\s*\(([-\d\.]+)\s*,\s*([-\d\.]+)\)"
    )

    preds = []
    gts = []
    current_pred = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Predicted GPS
            m_pred = pred_pattern.search(line)
            if m_pred:
                lat = float(m_pred.group(1))
                lon = float(m_pred.group(2))
                current_pred = (lat, lon)
                continue

            # Ground Truth GPS
            m_gt = gt_pattern.search(line)
            if m_gt:
                lat = float(m_gt.group(1))
                lon = float(m_gt.group(2))
                if current_pred is not None:
                    preds.append(current_pred)
                    gts.append((lat, lon))
                    current_pred = None

    if not preds or not gts:
        raise ValueError("No prediction / ground truth pairs found in log.")

    pred_arr = np.array(preds, dtype=float)
    gt_arr = np.array(gts, dtype=float)

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Pred and GT arrays have different shapes: "
            f"{pred_arr.shape} vs {gt_arr.shape}"
        )

    return pred_arr, gt_arr


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance (km) between two lat/lon points using haversine.
    Inputs in degrees.
    """
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    c = 2.0 * math.asin(math.sqrt(a))

    return R * c


def compute_distances(pred, gt):
    """
    pred, gt: (N, 2) arrays [lat, lon]
    Returns:
        distances: (N,) array of haversine distances in km
    """
    distances = []
    for (plat, plon), (glat, glon) in zip(pred, gt):
        d = haversine_km(plat, plon, glat, glon)
        distances.append(d)
    return np.array(distances, dtype=float)


def plot_histogram(distances, output_prefix=None):
    counts, bins = np.histogram(distances, bins=100)
    width = (bins[1] - bins[0]) * 0.8     # 80% width → 20% gap
    centers = (bins[:-1] + bins[1:]) / 2
    
    plt.figure()
    plt.bar(centers, counts, width=width, edgecolor='black')
    plt.xlabel("Distance Error (km)")
    plt.ylabel("Count")
    plt.title("Histogram of Prediction Errors")

    if output_prefix:
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_error_KDE_histogram.png", dpi=200)
        plt.close()
    else:
        plt.show()
        
def plot_histogram2(distances, distances2, output_prefix=None):
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    
    counts, bins = np.histogram(distances, bins=100)
    width = (bins[1] - bins[0]) * 0.8     # 80% width → 20% gap
    centers = (bins[:-1] + bins[1:]) / 2
    ax[0].bar(centers, counts, width=width, edgecolor='black')
    ax[0].set_xlabel("Distance Error (km)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Histogram of Prediction Errors (ResNet)")
    
    counts2, bins2 = np.histogram(distances2, bins=100)
    centers2 = (bins2[:-1] + bins2[1:]) / 2
    ax[1].bar(centers2, counts2, width=width, edgecolor='black', color='orange')
    ax[1].set_xlabel("Distance Error (km)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Histogram of Prediction Errors (CLIP)")

    if output_prefix:
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_error_KDE_histogram.png", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_pairs_world(pred, gt, indices, title, output_prefix=None, suffix=""):
    """
    Plot selected prediction/GT pairs on a world map.

    - World basemap from Natural Earth via GeoPandas.
    - Each pair uses a unique marker color (no red/green).
    - GT bounds (circles) are green.
    - Pred bounds (circles) are red.
    """

    if len(indices) == 0:
        return

    # Load world map (EPSG:4326)
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    # Colors for pairs (avoid red/green)
    pair_colors = [
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "k",
        "m",
    ]

    # Compute a reasonable radius in degrees for bounds
    gts_sel = gt[indices]
    preds_sel = pred[indices]
    all_lats = np.concatenate([gts_sel[:, 0], preds_sel[:, 0]])
    all_lons = np.concatenate([gts_sel[:, 1], preds_sel[:, 1]])

    lat_span = float(all_lats.max() - all_lats.min()) if len(all_lats) > 0 else 1.0
    lon_span = float(all_lons.max() - all_lons.min()) if len(all_lons) > 0 else 1.0
    span = max(lat_span, lon_span)
    # Smallish fraction of the global span
    radius = 0.03 * span if span > 0 else 1.0  # degrees

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot world map
    world.plot(ax=ax, edgecolor="black", facecolor="lightgray")

    for idx_in_list, i in enumerate(indices):
        color = pair_colors[idx_in_list % len(pair_colors)]

        glat, glon = gt[i]
        plat, plon = pred[i]

        # Create one-row GeoDataFrames for plotting
        gt_gdf = gpd.GeoDataFrame(
            geometry=[Point(glon, glat)],
            crs="EPSG:4326",
        )
        pred_gdf = gpd.GeoDataFrame(
            geometry=[Point(plon, plat)],
            crs="EPSG:4326",
        )

        # Plot GT and Pred points (non-red/green colors)
        gt_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="x",
        )
        pred_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="^",
        )

        # Connect them with a line
        ax.plot(
            [glon, plon],
            [glat, plat],
            linestyle="--",
            linewidth=1.0,
            color=color,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if output_prefix:
        plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()
        
def plot_pairs_world2(pred, gt, pred2, gt2, indices, indices2, title, output_prefix=None, suffix=""):
    """
    Plot selected prediction/GT pairs on a world map.

    - World basemap from Natural Earth via GeoPandas.
    - Each pair uses a unique marker color (no red/green).
    - GT bounds (circles) are green.
    - Pred bounds (circles) are red.
    """

    if len(indices) == 0:
        return

    # Load world map (EPSG:4326)
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    # Colors for pairs (avoid red/green)
    # Ensure first dataset maps to ResNet (orange) and second to CLIP (blue)
    pair_colors = [
        "tab:orange",
        "tab:blue",
    ]

    # Compute a reasonable radius in degrees for bounds
    gts_sel = gt[indices]
    preds_sel = pred[indices]
    all_lats = np.concatenate([gts_sel[:, 0], preds_sel[:, 0]])
    all_lons = np.concatenate([gts_sel[:, 1], preds_sel[:, 1]])

    lat_span = float(all_lats.max() - all_lats.min()) if len(all_lats) > 0 else 1.0
    lon_span = float(all_lons.max() - all_lons.min()) if len(all_lons) > 0 else 1.0
    span = max(lat_span, lon_span)
    # Smallish fraction of the global span
    radius = 0.03 * span if span > 0 else 1.0  # degrees

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot world map
    world.plot(ax=ax, edgecolor="black", facecolor="lightgray")

    for idx_in_list, i in enumerate(indices):
        color = pair_colors[0]

        glat, glon = gt[i]
        plat, plon = pred[i]

        # Create one-row GeoDataFrames for plotting
        gt_gdf = gpd.GeoDataFrame(
            geometry=[Point(glon, glat)],
            crs="EPSG:4326",
        )
        pred_gdf = gpd.GeoDataFrame(
            geometry=[Point(plon, plat)],
            crs="EPSG:4326",
        )

        # Plot GT and Pred points (non-red/green colors)
        gt_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="x",
        )
        pred_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="^",
        )

        # Connect them with a line
        ax.plot(
            [glon, plon],
            [glat, plat],
            linestyle="--",
            linewidth=1.0,
            color=color,
        )
        
    for idx_in_list, i in enumerate(indices2):
        color = pair_colors[1]

        glat, glon = gt2[i]
        plat, plon = pred2[i]

        # Create one-row GeoDataFrames for plotting
        gt_gdf = gpd.GeoDataFrame(
            geometry=[Point(glon, glat)],
            crs="EPSG:4326",
        )
        pred_gdf = gpd.GeoDataFrame(
            geometry=[Point(plon, plat)],
            crs="EPSG:4326",
        )

        # Plot GT and Pred points (non-red/green colors)
        gt_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="x",
        )
        pred_gdf.plot(
            ax=ax,
            color=color,
            markersize=30,
            marker="^",
        )

        # Connect them with a line
        ax.plot(
            [glon, plon],
            [glat, plat],
            linestyle="--",
            linewidth=1.0,
            color=color,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")

    # Add legend explaining color <-> model and marker <-> role
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:orange', markersize=10, label='ResNet'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:blue', markersize=10, label='CLIP'),
        Line2D([0], [0], marker='^', color='k', markersize=10, linestyle='None', label='Prediction'),
        Line2D([0], [0], marker='x', color='k', markersize=10, linestyle='None', label='Ground Truth'),
    ]

    ax.legend(handles=legend_elements, loc='lower left')
    plt.tight_layout()

    if output_prefix:
        plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute GPS prediction errors and plot histogram + closest/furthest pairs on a world map."
    )
    parser.add_argument("--log_file", help="Path to the .log file to read")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="If given, save plots to files with this prefix instead of showing them.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many closest/furthest pairs to plot (default: 10)",
    )

    args = parser.parse_args()

    pred, gt = parse_log_file(args.log_file)
    distances = compute_distances(pred, gt)

    # Print some stats
    print(f"Number of pairs: {len(distances)}")
    print("Distances (km):")
    for i, d in enumerate(distances):
        print(f"  Pair {i}: {d:.3f} km")

    print(f"\nMean error: {distances.mean():.3f} km")
    print(f"Median error: {np.median(distances):.3f} km")
    print(f"Min error: {distances.min():.3f} km")
    print(f"Max error: {distances.max():.3f} km")

    # Histogram
    plot_histogram(distances, output_prefix=args.output_prefix)

    # Top-K closest / furthest
    sorted_idx = np.argsort(distances)
    k = min(args.top_k, len(sorted_idx))
    closest_idx = sorted_idx[:k]
    furthest_idx = sorted_idx[-k:]
    print(len(distances))

    # Closest pairs map
    plot_pairs_world(
        pred,
        gt,
        closest_idx,
        title=f"Top {k} Closest Prediction–GT Pairs (World Map)",
        output_prefix=args.output_prefix,
        suffix="closest_pairs_world",
    )

    # Furthest pairs map
    plot_pairs_world(
        pred,
        gt,
        furthest_idx,
        title=f"Top {k} Furthest Prediction–GT Pairs (World Map)",
        output_prefix=args.output_prefix,
        suffix="furthest_pairs_world",
    )
    
def main2():
    parser = argparse.ArgumentParser(
        description="Compute GPS prediction errors and plot histogram + closest/furthest pairs on a world map."
    )
    parser.add_argument("--log_file", help="Path to the .log file to read")
    parser.add_argument("--log_file2", help="Path to the .log file to read")
    
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="If given, save plots to files with this prefix instead of showing them.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many closest/furthest pairs to plot (default: 10)",
    )

    args = parser.parse_args()

    pred, gt = parse_log_file(args.log_file)
    pred2, gt2 = parse_log_file(args.log_file2)
    distances = compute_distances(pred, gt)
    distances2 = compute_distances(pred2, gt2)

    # Print some stats
    print(f"Number of pairs for logfile1: {len(distances)}")
    print(f"Number of pairs for logfile2: {len(distances2)}")
    
    # print("Distances (km):")
    # for i, d in enumerate(distances):
    #     print(f"  Pair {i}: {d:.3f} km")

    print(f"\nMean error for logfile1: {distances.mean():.3f} km")
    print(f"Median error for logfile1: {np.median(distances):.3f} km")
    print(f"Min error for logfile1: {distances.min():.3f} km")
    print(f"Max error for logfile1: {distances.max():.3f} km")

    print(f"\nMean error for logfile2: {distances2.mean():.3f} km")
    print(f"Median error for logfile2: {np.median(distances2):.3f} km")
    print(f"Min error for logfile2: {distances2.min():.3f} km")
    print(f"Max error for logfile2: {distances2.max():.3f} km")

    # # Histogram
    # plot_histogram(distances, output_prefix=args.output_prefix)
    
    plot_histogram2(distances, distances2, output_prefix=args.output_prefix)

    # Top-K closest / furthest
    sorted_idx = np.argsort(distances)
    k = min(args.top_k, len(sorted_idx))
    closest_idx = sorted_idx[:k]
    furthest_idx = sorted_idx[-k:]
    
    sorted_idx2 = np.argsort(distances2)
    k = min(args.top_k, len(sorted_idx2))
    closest_idx2 = sorted_idx2[:k]
    furthest_idx2 = sorted_idx2[-k:]
    plot_pairs_world2(
        pred,
        gt,
        pred2,
        gt2,
        closest_idx,
        closest_idx2,
        title=f"Top {k} Closest Prediction–GT Pairs (World Map)",
        output_prefix=args.output_prefix,
        suffix="closest_pairs_world",
    )
    
    plot_pairs_world2(
        pred,
        gt,
        pred2,
        gt2,
        furthest_idx,
        furthest_idx2,
        title=f"Top {k} Furthest Prediction–GT Pairs (World Map)",
        output_prefix=args.output_prefix,
        suffix="furthest_pairs_world",
    )

    # # Closest pairs map
    # plot_pairs_world(
    #     pred,
    #     gt,
    #     closest_idx,
    #     title=f"Top {k} Closest Prediction–GT Pairs (World Map)",
    #     output_prefix=args.output_prefix,
    #     suffix="closest_pairs_world",
    # )

    # # Furthest pairs map
    # plot_pairs_world(
    #     pred,
    #     gt,
    #     furthest_idx,
    #     title=f"Top {k} Furthest Prediction–GT Pairs (World Map)",
    #     output_prefix=args.output_prefix,
    #     suffix="furthest_pairs_world",
    # )


if __name__ == "__main__":
    main2()
