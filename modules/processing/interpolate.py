from scipy.interpolate import griddata
import alphashape
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt

def interpolate_data(df_sample, var, nx=500, ny=500, alpha=0.0001):
    grid_x, grid_y = np.meshgrid(np.linspace(df_sample['X'].min(), df_sample['X'].max(), nx), np.linspace(df_sample['Y'].min(), df_sample['Y'].max(), ny))
    data_interpolated = griddata(np.column_stack((df_sample['X'], df_sample['Y'])), df_sample[var], (grid_x.ravel(), grid_y.ravel()), method='cubic')

    shape = alphashape.alphashape(np.column_stack((df_sample['X'], df_sample['Y'])), alpha=alpha)

    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    inside_mask = np.array([shape.contains(Point(p)) for p in points])

    data_masked = np.where(inside_mask, data_interpolated, np.nan)
    return grid_x, grid_y, data_interpolated, data_masked, inside_mask

def plot_interpolated_data(df_sample, grid_x, grid_y, data_interpolated, data_masked, inside_mask, var, cmap='viridis'):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 5))

    vmin = df_sample[var].min()
    vmax = df_sample[var].max()
    im0 = ax0.scatter(df_sample['X'], df_sample['Y'], c=df_sample[var], s=10, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im0, ax=ax0)
    cb.set_label(var, fontsize=12)
    ax0.set_title(f'{var} values', fontsize=14)
    ax0.set_xlabel("Easting (m)", fontsize=12)
    ax0.set_ylabel("Northing (m)", fontsize=12)
    ax0.set_aspect('equal', adjustable='box')
    ax0.locator_params(axis='x', nbins=5)
    ax0.locator_params(axis='y', nbins=5)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(1,5))

    im1 = ax1.pcolormesh(grid_x, grid_y, data_interpolated.reshape((grid_x.shape)), cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im1, ax=ax1)
    cb.set_label(var, fontsize=12)
    ax1.set_title(f'{var} values', fontsize=14)
    ax1.set_xlabel("Easting (m)", fontsize=12)
    ax1.set_ylabel("Northing (m)", fontsize=12)
    ax1.set_aspect('equal', adjustable='box')
    ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=5)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(1,5))
        
    im2 = ax2.pcolormesh(grid_x, grid_y, inside_mask.reshape((grid_x.shape)))
    ax2.set_title('Máscara', fontsize=14)
    ax2.set_xlabel("Easting (m)", fontsize=12)
    ax2.set_ylabel("Northing (m)", fontsize=12)
    ax2.set_aspect('equal', adjustable='box')
    ax2.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='y', nbins=5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(1,5))

    im3 = ax3.pcolormesh(grid_x, grid_y, data_masked.reshape((grid_x.shape)), cmap=cmap)
    cb = plt.colorbar(im3, ax=ax3)
    cb.set_label(var, fontsize=12)
    ax3.set_title(f'{var} values', fontsize=14)
    ax3.set_xlabel("Easting (m)", fontsize=12)
    ax3.set_ylabel("Northing (m)", fontsize=12)
    ax3.set_aspect('equal', adjustable='box')
    ax3.locator_params(axis='x', nbins=5)
    ax3.locator_params(axis='y', nbins=5)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(1,5))

    fig.tight_layout()
    plt.show()