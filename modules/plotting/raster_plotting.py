import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def custom_cmap(cmap_name):

    if cmap_name == 'terrain_crust':
        cmap_terrain = plt.get_cmap('terrain')

        cmap_terrain_crust = LinearSegmentedColormap.from_list(
            'terrain_crust', cmap_terrain(np.linspace(0.25, 1, 256))
        )
        cmap_terrain_crust.set_bad(color='blue')
        cmap = cmap_terrain_crust

    elif cmap_name == 'multiple_colors':
        cmap_tab10 = cm.get_cmap("tab10", 10)
        colors = [cmap_tab10(i) for i in range(10)]

        colors_with_white = []
        for c in colors:
            colors_with_white.append(c)
            colors_with_white.append("white")

        colors_with_white = colors_with_white[:-1]

        multiple_colors = mcolors.LinearSegmentedColormap.from_list("tab10_with_white", colors_with_white, N=256)
        cmap = multiple_colors

    elif cmap_name == 'GA':
        colors = [
            "#0919c4",
            "#07e6ff",
            "#f3ff0e",
            "#ff0000"
        ]
        cmap_GA = mcolors.LinearSegmentedColormap.from_list("divergente_centro_blanco", colors, N=256)
        cmap = cmap_GA

    return cmap

def get_plot_config(data_type, **user_params):
    configs = {
        'DEM': {
            'title': "Digital Elevation Model",
            'data_title': "Elevation (m)",
            'cmap': custom_cmap('terrain_crust'),
            'vmin': 0,
            'vmax': 4500
        },
        'TMI': {
            'title': "Total Magnetic Intensity",
            'data_title': "Intensity (nT)",
            'cmap': custom_cmap('multiple_colors'),
            'vmin': None,
            'vmax': None
        },
        'GA': {
            'title': "Gravity Anomaly",
            'data_title': "Anomaly (mGal)",
            'cmap': custom_cmap('GA'),
            'vmin': None,
            'vmax': None
        }
    }
    
    default_config = {
        'title': "Raster Data",
        'data_title': "Value", 
        'cmap': 'gray',
        'figsize': (10,8),
        'vmin': None,
        'vmax': None
    }
    
    config = {**default_config, **configs.get(data_type, {}), **user_params}
    return config

def plot_raster_data(x, y, z, data_type=None, **kwargs):
    config = get_plot_config(data_type, **kwargs)
    
    if data_type == 'DEM':
        z = np.ma.masked_where(z == 0, z)

    elif data_type == 'GA':
        vmax = np.nanmax(np.abs(z))
        config['vmax'] = vmax
        config['vmin'] = -vmax

    fig, ax = plt.subplots(figsize=config['figsize'])
    im = ax.pcolormesh(x, y, z, cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'])
    cb = plt.colorbar(im)
    cb.set_label(config['data_title'], fontsize=12)
    ax.set_title(config['title'], fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(1,5))
    fig.tight_layout()

def plot_scatter_data(x, y, z, data_type=None, **kwargs):
    config = get_plot_config(data_type, **kwargs)
    
    if data_type == 'DEM':
        z = np.ma.masked_where(z == 0, z)

    fig, ax = plt.subplots(figsize=config['figsize'])
    sc = ax.scatter(x, y, c=z.ravel(), cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'], s=10)
    cb = plt.colorbar(sc)
    cb.set_label(config['data_title'], fontsize=12)
    ax.set_title(config['title'], fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(1,5))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(1,5))
    fig.tight_layout()