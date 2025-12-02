import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load CSV
df = pd.read_csv("filtered_images.csv")   # replace with your filename

# 2. Convert to GeoDataFrame
points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
    crs="EPSG:4326"   # WGS84
)

# 3. Load a world map background (optional)
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

# 4. Plot
ax = world.plot(figsize=(10, 5), edgecolor="black")
points.plot(ax=ax, color="red", markersize=1)

plt.title("All Points from CSV")
plt.tight_layout()
plt.savefig("points_map.png", dpi=300)
