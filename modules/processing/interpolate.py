from scipy.interpolate import griddata
import alphashape
from shapely.geometry import Point
import numpy as np
import verde as vd

def interpolate_data(df_sample, var, coords_type = ('x', 'y'), nx=500, ny=500, alpha=0.0001, return_shape=False, x_values=None, y_values=None, **kwargs):
    if x_values is not None and y_values is not None:
        grid_x, grid_y = np.meshgrid(x_values, y_values)
    else:
        grid_x, grid_y = np.meshgrid(np.linspace(df_sample[coords_type[0]].values.min(), df_sample[coords_type[0]].values.max(), nx), np.linspace(df_sample[coords_type[1]].values.min(), df_sample[coords_type[1]].values.max(), ny))
    data_interpolated = griddata(np.column_stack((df_sample[coords_type[0]].values, df_sample[coords_type[1]].values)), df_sample[var].values, (grid_x.ravel(), grid_y.ravel()), method='cubic')

    shape = alphashape.alphashape(np.column_stack((df_sample[coords_type[0]].values, df_sample[coords_type[1]].values)), alpha=alpha)

    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    inside_mask = np.array([shape.contains(Point(p)) for p in points])

    data_masked = np.where(inside_mask, data_interpolated, np.nan)
    if return_shape:
        return grid_x, grid_y, data_interpolated, data_masked, inside_mask, shape
    else:
        return grid_x, grid_y, data_interpolated, data_masked, inside_mask

def interpolate_data_spline(data, y_values, x_values, sampling_factor=100):
    """
    Interpolates data using Verde's Spline with prior subsampling.
    
    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Dataset with data containing 'lat', 'long', and 'anomaly'
    y_values : array-like
        Grid values on the Y axis (latitude)
    x_values : array-like
        Grid values on the X axis (longitude)
    sampling_factor : int, optional
        Subsampling factor (default=100)
    
    Returns
    -------
    values_interpolated : np.ndarray
        Interpolated values on the full grid
    """
    
    # Subsampling
    data_subsampled = data.isel(index=slice(None, None, sampling_factor))
    
    coords_valid = (data_subsampled.lat.values, data_subsampled.long.values)
    values_valid = data_subsampled['anomaly'].values

    print(f"Number of points in data grid: {data['anomaly'].size}")
    print(f"Number of valid points for spline fitting: {len(values_valid)}")
    
    # Create spline with subsampled data
    spline = vd.Spline()
    spline.fit(coords_valid, values_valid)
    
    # Interpolate on full grid
    y_full, x_full = np.meshgrid(y_values, x_values, indexing='ij')
    
    coords_to_predict = (y_full, x_full)
    values_interpolated = spline.predict(coords_to_predict)
    
    return values_interpolated