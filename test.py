# Instalación: pip install verde harmonica boule rasterio torch torchvision scikit-image shapely geopandas
import os
from pathlib import Path


def _configure_conda_mkl_dlls() -> None:
    """Ensure MKL/OpenMP DLL folders are available when launching from Git Bash."""
    if os.name != "nt":
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                Path(conda_prefix) / "Library" / "bin",
                Path(conda_prefix) / "DLLs",
            ]
        )

    # Fallback for direct execution with an env python.exe from Git Bash.
    py_root = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            py_root / "Library" / "bin",
            py_root / "DLLs",
            py_root.parent / "Library" / "bin",  # conda base often contains versioned MKL DLLs
        ]
    )

    for folder in candidates:
        if folder.is_dir():
            try:
                os.add_dll_directory(str(folder))
            except (AttributeError, FileNotFoundError, OSError):
                pass


import sys

_configure_conda_mkl_dlls()

import verde as vd
import harmonica as hm
import boule as bl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_dilation
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. PREPROCESAMIENTO CON VERDE (datos reales o ejemplo Baja California)
raw_data = vd.datasets.fetch_baja_bathymetry()  # Reemplaza con tus datos: vd.load_surfer('pacifico_dem.grd')
scatter_coords = (raw_data.longitude.values, raw_data.latitude.values)
dem_data = raw_data.bathymetry_m.values
region = vd.get_region(scatter_coords)

# Reducción en bloque y grillado
reducer = vd.BlockMean(spacing=0.05, region=region)
coords_reduced, values_reduced, _ = reducer.filter(scatter_coords, dem_data)
gridder = vd.Linear()
gridder.fit(coords_reduced, values_reduced)
grid_dataset = gridder.grid(region=region, spacing=0.05, data_names=["bathymetry"])
dem_grid = grid_dataset.bathymetry  # xarray.DataArray (northing, easting)

# Stack multi-canal: DEM, corrección Bouguer, mag total, gradientes, tilt
dem_filled = dem_grid.fillna(0)  # FFT-based transforms require no NaN values
bouguer_corr = hm.bouguer_correction(dem_filled.values, density_crust=2670)
lat_2d = dem_filled.northing.values[:, None] * np.ones((1, dem_filled.shape[1]))
normal_grav = bl.WGS84.normal_gravity(latitude=lat_2d, height=np.zeros_like(dem_filled.values))
anomaly = bouguer_corr - normal_grav

# Derivados (tilt angle para lineamientos) sobre xarray.DataArray
gx = hm.derivative_easting(dem_filled)
gy = hm.derivative_northing(dem_filled)
tilt = hm.tilt_angle(dem_filled)

# Grilla magnética (placeholder — reemplaza con datos reales si disponibles)
magnetic_grid = np.zeros_like(dem_filled.values)

# Stack: (H, W, 6 canales)
stacked = np.stack([dem_filled.values, anomaly, magnetic_grid, gx.values, gy.values, tilt.values], axis=-1)

# 2. GENERAR SINTÉTICOS CON HARMONICA (para labels/training)
# Franja central como falla dike-like sintética
fault_mask = np.zeros(dem_filled.shape, dtype=bool)
h, w = dem_filled.shape
fault_mask[h // 3:2 * h // 3, w // 4:3 * w // 4] = True
synthetic_mask = binary_dilation(remove_small_objects(fault_mask, min_size=10))

# 3. DATASET PARA U-NET
class GeoLineamentDataset(Dataset):
    def __init__(self, stacked_rasters, mask_data):
        self.data = torch.tensor(stacked_rasters, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        self.labels = torch.tensor(mask_data.astype(np.float32)).unsqueeze(0)  # Binary mask
    
    def __len__(self):
        return 1  # Batch por raster; augmenta con Verde para más
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = GeoLineamentDataset(stacked, synthetic_mask)
dataloader = DataLoader(dataset, batch_size=1)

# 4. U-NET SIMPLE (segmentación lineamientos)
class UNetLineament(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        # ... más capas (encoder-decoder, típico U-Net 4 niveles)
        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU())  # Upsample
        self.final = nn.Conv2d(64, out_channels, 1)  # Binary output
    
    def forward(self, x):
        # Encoder-decoder con skips
        return torch.sigmoid(self.final(x))  # Probabilidad

model = UNetLineament().cuda() if torch.cuda.is_available() else UNetLineament()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Training loop (50 epochs)
for epoch in range(50):
    for imgs, masks_batch in dataloader:
        preds = model(imgs.cuda())
        loss = criterion(preds, masks_batch.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# 5. INFERENCIA + VECTORIZACIÓN
def predict_lineaments(trained_model, raster_path):
    with rasterio.open(raster_path) as src:  # Input: multi-canal GeoTIFF
        img = src.read()  # (C,H,W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    pred = trained_model(img_tensor).squeeze().detach().numpy() > 0.5  # Threshold
    skel = skeletonize(pred)  # Líneas delgadas
    return skel

skeleton = predict_lineaments(model, 'pacifico_stack.tif')

# Export shapefile (vectores)
lines = []  # Detecta líneas con Hough o connected components
gdf = gpd.GeoDataFrame(lines, crs='EPSG:4326')
gdf.to_file('lineaments.shp')

# Visualiza
plt.imshow(skeleton)
plt.show()