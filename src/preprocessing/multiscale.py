import numpy as np
from scipy.ndimage import zoom


def _resize_to_shape(arr, target_shape, order=1):
    scale_y = target_shape[0] / arr.shape[0]
    scale_x = target_shape[1] / arr.shape[1]
    return zoom(arr, (scale_y, scale_x), order=order)


def build_multiscale_feature_pyramid(feature_2d, scales=(1, 2, 4), order=1):
    """
    Build a multiscale pyramid and upsample all levels to the original shape.

    Parameters
    ----------
    feature_2d : ndarray (H, W)
    scales : tuple[int]
        Scale factors. 1 means native resolution, 2 means half-res, etc.
    order : int
        Interpolation order passed to scipy.ndimage.zoom.
    """
    base = np.asarray(feature_2d).astype(float)
    h, w = base.shape

    levels = {}
    for scale in scales:
        if scale < 1:
            raise ValueError("All scales must be >= 1")
        if scale == 1:
            levels[f"x{scale}"] = base
            continue

        down_h = max(2, h // scale)
        down_w = max(2, w // scale)
        down = _resize_to_shape(base, (down_h, down_w), order=order)
        up = _resize_to_shape(down, (h, w), order=order)
        levels[f"x{scale}"] = up

    return levels


def append_multiscale_channels(feature_dict, target_keys, scales=(1, 2, 4), order=1):
    """
    Add multiscale versions of selected features to the feature dictionary.

    Returns
    -------
    dict
        Copy of feature_dict with extra entries named <feature>_x<scale>.
    """
    out = dict(feature_dict)
    for key in target_keys:
        if key not in feature_dict:
            raise KeyError(f"Feature '{key}' is missing from feature_dict")
        pyr = build_multiscale_feature_pyramid(feature_dict[key], scales=scales, order=order)
        for level_name, arr in pyr.items():
            out[f"{key}_{level_name}"] = arr
    return out
