from .geophysical_features import (
    build_multisource_feature_stack,
    compute_analytic_signal_features,
    compute_dem_curvatures,
    compute_dem_derivatives,
    compute_directional_derivatives,
    compute_gabor_features,
    compute_glcm_spatial_features,
    compute_tga_tilt,
    reduction_to_pole,
    separate_regional_residual,
)
from .multiscale import (
    append_multiscale_channels,
    build_multiscale_feature_pyramid,
)
from .prepare_multisource_data import (
    prepare_multisource_dataset_from_directories,
    prepare_single_multisource_sample,
)
from .rasterized_training_data import (
    rasterize_line_vectors_to_mask,
    tile_prestacked_raster_and_mask,
)

try:
    from .multisource_dataset import MultiSourceStackDataset
except ModuleNotFoundError:  # torch is optional for non-training workflows
    MultiSourceStackDataset = None

__all__ = [
    "build_multisource_feature_stack",
    "compute_analytic_signal_features",
    "compute_dem_curvatures",
    "compute_dem_derivatives",
    "compute_directional_derivatives",
    "compute_gabor_features",
    "compute_glcm_spatial_features",
    "compute_tga_tilt",
    "reduction_to_pole",
    "separate_regional_residual",
    "append_multiscale_channels",
    "build_multiscale_feature_pyramid",
    "prepare_multisource_dataset_from_directories",
    "prepare_single_multisource_sample",
    "rasterize_line_vectors_to_mask",
    "tile_prestacked_raster_and_mask",
]

if MultiSourceStackDataset is not None:
    __all__.append("MultiSourceStackDataset")
