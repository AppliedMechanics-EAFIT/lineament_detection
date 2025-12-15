import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

def read_ascii_xyz(file_path, line_str, var_types, coords_type = ('LONG', 'LAT')):
    """
    Read ASCII XYZ file and return an xarray Dataset with lat/long coordinates sorted by longitude.
    
    Parameters
    ----------
    filename : str
        Path to the ASCII XYZ file
    line_str : str
        String identifying the start of each line (e.g., 'line')
    var_types : dict
        Dictionary with data types for each column
    coords_type : tuple, optional
        Tuple with the names of the coordinate columns (default is ('LONG', 'LAT'))

    Returns
    -------
    xarray.Dataset
        Dataset with lat/long coordinates sorted by longitude

    Examples
    --------
    >>> import os
    >>> data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    >>> line_str = 'line'

    >>> var_types = {
    ...     'lat': float,
    ...     'long': float,
    ...     'TMI': float,
    ...     'line': str
    ... }
    >>> file_path = os.path.join(data_dir, 'magnetic', 'mag_region_a.xyz')
    >>> ds = read_ascii_xyz(file_path, line_str, var_types)
    """

    blocks = []
    block_ids = []

    with open(file_path) as f:
        lines = f.readlines()

    columns = lines[1].strip().split()
    current_block = None

    for line in lines[2:]:
        if line.startswith(line_str):
            current_block = line.strip()
            continue
        values = line.strip().split()
        if len(values) == len(columns):
            blocks.append(values)
            block_ids.append(current_block)

    df = pd.DataFrame(blocks, columns=columns)
    df[line_str] = block_ids

    df = df.astype(var_types)
    ds = df.to_xarray()
    ds = ds.set_coords(coords_type).rename({k: k.lower() for k in coords_type}).sortby(coords_type[0].lower())
    return ds

def convert_time_to_seconds(time_str):
    """Convierte un string de tiempo HHMMSS.ss a segundos totales desde la medianoche."""
    try:
        # Asegura que el string tenga al menos 6 caracteres para HHMMSS
        time_str = time_str.split('.')[0].zfill(6)
        
        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return float(total_seconds)
    except (ValueError, IndexError):
        # Devuelve NaN si el formato es incorrecto
        return np.nan