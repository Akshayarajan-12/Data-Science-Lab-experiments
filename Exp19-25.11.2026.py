
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import Basemap
try:
    from mpl_toolkits.basemap import Basemap
    basemap_available = True
except ImportError:
    basemap_available = False
    print("Basemap is not installed. Please install it.")

# --------------------------------
# STEP 1: Create City Dataset
# --------------------------------

records = [
    ("Delhi", 28.61, 77.21, 122, 55, 12),
    ("Mumbai", 19.08, 72.88, 45, 70, 10),
    ("Kolkata", 22.57, 88.36, 40, 60, 8),
    ("Chennai", 13.08, 80.27, 62, 58, 14),
    ("Bengaluru", 12.97, 77.59, 105, 45, 16),
    ("Hyderabad", 17.39, 78.49, 77, 50, 12),
    ("Ahmedabad", 23.02, 72.57, 45, 75, 9),
    ("Pune", 18.52, 73.86, 40, 52, 15),
    ("Jaipur", 26.91, 75.79, 30, 30, 11),
    ("Lucknow", 26.85, 80.95, 28, 28, 10),
    ("Kanpur", 26.45, 80.33, 17, 48, 7),
    ("Patna", 25.59, 85.14, 18, 22, 6),
    ("Bhopal", 23.26, 77.41, 16, 25, 21),
    ("Kochi", 9.93, 76.27, 10, 20, 30),
    ("Visakhapatnam", 17.69, 83.22, 9, 40, 25),
    ("Guwahati", 26.14, 91.74, 11, 15, 28)
]

columns = [
    "city", "lat", "lon",
    "vehicles_lakh", "industries", "green_cover"
]

cities = pd.DataFrame(records, columns=columns)

# Generate PM2.5 values
np.random.seed(2)

cities["pm25"] = (
    25
    + 0.4 * cities["vehicles_lakh"]
    + 0.5 * cities["industries"]
    - 0.9 * cities["green_cover"]
    + np.random.normal(0, 5, len(cities))
).clip(lower=10).round(1)

# Add pollution level for Indo-Gangetic cities
belt = cities["city"].isin(["Kanpur", "Patna", "Lucknow"])
cities.loc[belt, "pm25"] += 40

# --------------------------------
# STEP 2: Display Dataset
# --------------------------------

print("\nAIR QUALITY ANALYSIS")
print("=" * 60)

print(cities[
    ["city", "vehicles_lakh", "industries",
     "green_cover", "pm25"]
].to_string(index=False))

# --------------------------------
# STEP 3: Correlation Analysis
# --------------------------------

corr = cities[
    ["vehicles_lakh", "industries",
     "green_cover", "pm25"]
].corr()

print("\nCORRELATION MATRIX")
print(corr.round(3))

for col in ["vehicles_lakh", "industries", "green_cover"]:

    r = corr.loc[col, "pm25"]

    if abs(r) > 0.6:
        strength = "Strong"
    elif abs(r) > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"

    direction = "Positive" if r > 0 else "Negative"

    print(
        f"PM2.5 vs {col}: "
        f"{r:.3f} ({strength} {direction})"
    )

print("\nMost polluted cities:")
print(cities.nlargest(3, "pm25")["city"].tolist())

print("\nLeast polluted cities:")
print(cities.nsmallest(3, "pm25")["city"].tolist())

# --------------------------------
# STEP 4: Scatter Plot
# --------------------------------

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

scatter = ax[0].scatter(
    cities["vehicles_lakh"],
    cities["pm25"],
    s=cities["industries"] * 6,
    c=cities["green_cover"],
    cmap="RdYlGn",
    edgecolors="black",
    alpha=0.8
)

# Fit a linear regression line
m, b = np.polyfit(
    cities["vehicles_lakh"],
    cities["pm25"],
    1
)

xs = np.linspace(0, 130, 50)

ax[0].plot(
    xs,
    m * xs + b,
    "k--",
    label="Fitted Trend Line"
)

# Add city labels
for _, row in cities.iterrows():
    ax[0].annotate(
        row["city"],
        (row["vehicles_lakh"], row["pm25"]),
        fontsize=7,
        xytext=(4, 4),
        textcoords="offset points"
    )

fig.colorbar(
    scatter,
    ax=ax[0],
    label="Green Cover (%)"
)

ax[0].set_xlabel("Registered Vehicles (Lakh)")
ax[0].set_ylabel("PM2.5 (ug/m3)")
ax[0].set_title("Vehicles vs PM2.5")
ax[0].legend()
ax[0].grid(alpha=0.3)

# --------------------------------
# STEP 5: Correlation Heatmap
# --------------------------------

im = ax[1].imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

labels = [
    "Vehicles",
    "Industries",
    "Green Cover",
    "PM2.5"
]

ax[1].set_xticks(range(4))
ax[1].set_xticklabels(labels, rotation=30)

ax[1].set_yticks(range(4))
ax[1].set_yticklabels(labels)

for i in range(4):
    for j in range(4):
        ax[1].text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax[1])
ax[1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("exp19_scatter.png", dpi=150)

# --------------------------------
# STEP 6: Geographic Map of India
# --------------------------------

if basemap_available:

    plt.figure(figsize=(9, 9))

    mp = Basemap(
        projection="merc",
        llcrnrlat=6,
        urcrnrlat=36,
        llcrnrlon=67,
        urcrnrlon=98,
        resolution="l"
    )

    mp.drawcoastlines(linewidth=0.6)
    mp.drawcountries(linewidth=0.8)

    mp.fillcontinents(
        color="#f2efe6",
        lake_color="#cfe8f7"
    )

    mp.drawmapboundary(
        fill_color="#cfe8f7"
    )

    mp.drawparallels(
        range(5, 40, 5),
        labels=[1, 0, 0, 0],
        fontsize=8
    )

    mp.drawmeridians(
        range(65, 100, 5),
        labels=[0, 0, 0, 1],
        fontsize=8
    )

    x, y = mp(
        cities["lon"].values,
        cities["lat"].values
    )

    points = mp.scatter(
        x,
        y,
        s=cities["pm25"] * 4,
        c=cities["pm25"],
        cmap="YlOrRd",
        edgecolors="black",
        zorder=5
    )

    for xi, yi, name in zip(
        x, y, cities["city"]
    ):
        plt.text(
            xi + 30000,
            yi + 30000,
            name,
            fontsize=8,
            zorder=6
        )

    plt.colorbar(
        points,
        shrink=0.6,
        label="PM2.5 (ug/m3)"
    )

    plt.title(
        "PM2.5 Levels Across Indian Cities"
    )

    plt.savefig(
        "exp19_map.png",
        dpi=150,
        bbox_inches="tight"
    )

    print("\nMap saved successfully.")

plt.show()

print("\nSaved Figures:")
print("1. exp19_scatter.png")
if basemap_available:
    print("2. exp19_map.png")

print("\nExperiment 19 Completed Successfully!")
