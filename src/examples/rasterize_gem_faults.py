import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SITES = {
    'california': {'bbox': [-122, 35, -117, 38]},
    'turkey': {'bbox': [29, 40, 36, 42]},
    'iceland': {'bbox': [-22, 63, -18, 65]}
}

def download_gem_faults(cache_path):
    """Download GEM active faults GeoJSON."""
    url = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"
    response = requests.get(url)
    response.raise_for_status()
    with open(cache_path, 'w') as f:
        f.write(response.text)

def main():
    parser = argparse.ArgumentParser(description="Rasterize GEM fault masks for sites")
    parser.add_argument("--site", required=True, choices=SITES.keys())
    args = parser.parse_args()
    
    site = args.site
    bbox = SITES[site]['bbox']
    
    base_dir = ROOT / 'data' / 'non_colombia' / site
    dem_path = base_dir / 'dem' / 'scene_001.tif'
    mask_dir = base_dir / 'mask'
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / 'scene_001.tif'
    
    # Load DEM to get grid
    with rasterio.open(dem_path) as dem_src:
        dem_profile = dem_src.profile
        dem_bounds = dem_src.bounds
        dem_crs = dem_src.crs
        height, width = dem_src.height, dem_src.width
    
    # Download faults
    cache_path = ROOT / 'tmp' / 'gem_active_faults.geojson'
    cache_path.parent.mkdir(exist_ok=True)
    if not cache_path.exists():
        print("Downloading GEM faults")
        download_gem_faults(cache_path)
    
    # Load and filter
    faults = gpd.read_file(cache_path)
    # Filter to bbox
    geom_bbox = box(*bbox)
    faults_filtered = faults[faults.intersects(geom_bbox)]
    
    if faults_filtered.empty:
        print("No faults in bbox")
        # Create empty mask
        mask_array = np.zeros((height, width), dtype='uint8')
    else:
        # Buffer faults by 1-2 pixels
        pixel_size = dem_profile['transform'][0]  # approx
        buffer_dist = 1.5 * pixel_size  # 1.5 px
        
        faults_buffered = faults_filtered.buffer(buffer_dist)
        
        # Rasterize
        shapes = [(geom, 1) for geom in faults_buffered.geometry]
        mask_array = rasterize(
            shapes,
            out_shape=(height, width),
            transform=dem_profile['transform'],
            dtype='uint8'
        )
    
    # Save mask
    mask_profile = dem_profile.copy()
    mask_profile.update({
        'dtype': 'uint8',
        'count': 1,
        'nodata': 0
    })
    
    with rasterio.open(mask_path, 'w', **mask_profile) as dst:
        dst.write(mask_array, 1)
    
    print(f"Mask saved to {mask_path}")

if __name__ == "__main__":
    main()