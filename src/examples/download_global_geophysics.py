import argparse
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# AOIs por sitio
SITES = {
    'california': {'bbox': [-122, 35, -117, 38]},
    'turkey': {'bbox': [29, 40, 36, 42]},
    'iceland': {'bbox': [-22, 63, -18, 65]}
}

def download_copernicus_dem(bbox, out_path):
    """Download Copernicus GLO-30 DEM for bbox, mosaic and clip."""
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Find tiles: 1x1 degree
    lats = range(int(lat_min), int(lat_max) + 1)
    lons = range(int(lon_min), int(lon_max) + 1)
    
    sources = []
    for lat in lats:
        for lon in lons:
            if lon < 0:
                lon_str = f'W{abs(lon):03d}'
            else:
                lon_str = f'E{lon:03d}'
            lat_str = f'N{lat:02d}'
            
            url = f'https://copernicus-dem-30m.s3.amazonaws.com/30/{lat_str}/{lon_str}/Copernicus_DSM_COG_30_{lat_str}_{lon_str}_DEM.tif'
            
            # Download to temp
            temp_path = out_path.parent / f'temp_{lat}_{lon}.tif'
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                src = rasterio.open(temp_path)
                sources.append(src)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
    
    if not sources:
        raise ValueError("No DEM tiles downloaded")
    
    # Mosaic
    mosaic, out_trans = merge(sources)
    
    # Save
    out_meta = sources[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans,
        'crs': CRS.from_epsg(4326)
    })
    
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(mosaic[0], 1)
    
    # Close sources
    for src in sources:
        src.close()
    
    # Remove temps
    for lat in lats:
        for lon in lons:
            temp_path = out_path.parent / f'temp_{lat}_{lon}.tif'
            if temp_path.exists():
                temp_path.unlink()

def download_icgem_gravity(bbox, out_path):
    """Download WGM2012 Bouguer gravity from ICGEM."""
    lon_min, lat_min, lon_max, lat_max = bbox
    
    url = (
        "http://icgem.gfz-potsdam.de/tom_longtime_series/calc_grid_pontus.php?"
        f"model=WGM2012&lat_min={lat_min}&lat_max={lat_max}&lon_min={lon_min}&lon_max={lon_max}&"
        "step=0.0003&format=ascii"
    )
    
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse ASCII grid
    lines = response.text.split('\n')
    header = {}
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('ncols'):
            header['ncols'] = int(line.split()[1])
        elif line.startswith('nrows'):
            header['nrows'] = int(line.split()[1])
        elif line.startswith('xllcorner'):
            header['xllcorner'] = float(line.split()[1])
        elif line.startswith('yllcorner'):
            header['yllcorner'] = float(line.split()[1])
        elif line.startswith('cellsize'):
            header['cellsize'] = float(line.split()[1])
        elif line.startswith('NODATA_value'):
            header['nodata'] = float(line.split()[1])
        elif line.strip() == '':
            data_start = i + 1
            break
    
    data = []
    for line in lines[data_start:]:
        if line.strip():
            data.extend([float(x) for x in line.split()])
    
    array = np.array(data).reshape(header['nrows'], header['ncols'])
    array = np.flipud(array)  # ICGEM has y from top
    
    # Save as tif
    transform = from_bounds(
        header['xllcorner'], header['yllcorner'], 
        header['xllcorner'] + header['ncols'] * header['cellsize'],
        header['yllcorner'] + header['nrows'] * header['cellsize'],
        header['ncols'], header['nrows']
    )
    
    out_meta = {
        'driver': 'GTiff',
        'height': header['nrows'],
        'width': header['ncols'],
        'count': 1,
        'dtype': 'float32',
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'nodata': header['nodata']
    }
    
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(array.astype('float32'), 1)

def download_emag2_magnetic(bbox, out_path):
    """Download EMAG2v3 magnetic anomaly, clip and resample to 30m."""
    # Download NetCDF
    url = "https://www.ngdc.noaa.gov/geomag/EMAG2/EMAG2_V3_20170530.nc"
    temp_nc = out_path.parent / 'emag2_temp.nc'
    
    response = requests.get(url)
    response.raise_for_status()
    with open(temp_nc, 'wb') as f:
        f.write(response.content)
    
    # Open with xarray
    ds = xr.open_dataset(temp_nc)
    
    # EMAG2 is in lat/lon, but resolution ~2km
    # Clip to bbox
    lon_min, lat_min, lon_max, lat_max = bbox
    clipped = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    # Resample to 30m grid
    # First, create target grid
    res = 30 / 111000  # approx degrees
    lons_new = np.arange(lon_min, lon_max, res)
    lats_new = np.arange(lat_min, lat_max, res)
    
    clipped_resampled = clipped.interp(lon=lons_new, lat=lats_new, method='linear')
    
    # Save as tif
    array = clipped_resampled['EMAG2_V3'].values
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, len(lons_new), len(lats_new))
    
    out_meta = {
        'driver': 'GTiff',
        'height': len(lats_new),
        'width': len(lons_new),
        'count': 1,
        'dtype': 'float32',
        'crs': CRS.from_epsg(4326),
        'transform': transform
    }
    
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(array.astype('float32'), 1)
    
    temp_nc.unlink()

def main():
    parser = argparse.ArgumentParser(description="Download global geophysics data for sites")
    parser.add_argument("--site", required=True, choices=SITES.keys())
    args = parser.parse_args()
    
    site = args.site
    bbox = SITES[site]['bbox']
    
    base_dir = ROOT / 'data' / 'non_colombia' / site
    dem_dir = base_dir / 'dem'
    grav_dir = base_dir / 'grav'
    mag_dir = base_dir / 'mag'
    
    dem_dir.mkdir(parents=True, exist_ok=True)
    grav_dir.mkdir(parents=True, exist_ok=True)
    mag_dir.mkdir(parents=True, exist_ok=True)
    
    dem_path = dem_dir / 'scene_001.tif'
    grav_path = grav_dir / 'scene_001.tif'
    mag_path = mag_dir / 'scene_001.tif'
    
    print(f"Downloading DEM for {site}")
    download_copernicus_dem(bbox, dem_path)
    
    print(f"Downloading gravity for {site}")
    download_icgem_gravity(bbox, grav_path)
    
    print(f"Downloading magnetic for {site}")
    download_emag2_magnetic(bbox, mag_path)
    
    print("Done")

if __name__ == "__main__":
    main()