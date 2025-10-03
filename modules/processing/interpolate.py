from scipy.interpolate import griddata
import alphashape
from shapely.geometry import Point
import numpy as np

def interpolate_data(df_sample, var, coords_type = ('x', 'y'), nx=500, ny=500, alpha=0.0001):
    grid_x, grid_y = np.meshgrid(np.linspace(df_sample[coords_type[0]].values.min(), df_sample[coords_type[0]].values.max(), nx), np.linspace(df_sample[coords_type[1]].values.min(), df_sample[coords_type[1]].values.max(), ny))
    data_interpolated = griddata(np.column_stack((df_sample[coords_type[0]].values, df_sample[coords_type[1]].values)), df_sample[var].values, (grid_x.ravel(), grid_y.ravel()), method='cubic')

    shape = alphashape.alphashape(np.column_stack((df_sample[coords_type[0]].values, df_sample[coords_type[1]].values)), alpha=alpha)

    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    inside_mask = np.array([shape.contains(Point(p)) for p in points])

    data_masked = np.where(inside_mask, data_interpolated, np.nan)
    return grid_x, grid_y, data_interpolated, data_masked, inside_mask