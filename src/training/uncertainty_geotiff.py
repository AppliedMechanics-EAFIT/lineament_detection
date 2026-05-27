from pathlib import Path

import numpy as np

from .multisource_pipeline import predict_stack_with_uncertainty


def export_uncertainty_maps_to_geotiff(
    model,
    stack_hwc,
    reference_raster_path,
    output_dir,
    prefix="prediction",
    n_samples=30,
    activation="sigmoid",
    device="cpu",
    include_expected_entropy=False,
):
    """
    Predict and export uncertainty maps as GeoTIFF files.

    Parameters
    ----------
    model : torch.nn.Module
    stack_hwc : np.ndarray (H, W, C)
    reference_raster_path : str
        Raster used only to copy CRS/transform metadata.
    output_dir : str
        Destination folder for GeoTIFF outputs.
    """
    try:
        import rasterio
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("rasterio is required for GeoTIFF export") from exc

    outputs = predict_stack_with_uncertainty(
        model=model,
        stack_hwc=stack_hwc,
        n_samples=n_samples,
        activation=activation,
        device=device,
    )

    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()
        ref_shape = src.shape

    h, w = stack_hwc.shape[:2]
    if ref_shape != (h, w):
        raise ValueError(
            f"Reference raster shape {ref_shape} does not match stack shape {(h, w)}"
        )

    profile.update(dtype="float32", count=1, compress="lzw")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = ["mean", "std", "entropy", "mutual_information"]
    if include_expected_entropy:
        keys.append("expected_entropy")

    saved_paths = {}
    for key in keys:
        arr = np.asarray(outputs[key], dtype=np.float32)
        out_path = output_dir / f"{prefix}_{key}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        saved_paths[key] = str(out_path)

    return saved_paths
