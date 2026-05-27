import json
from pathlib import Path

import numpy as np

from .geophysical_features import build_multisource_feature_stack
from .multiscale import append_multiscale_channels


def _read_single_band_raster(path):
    try:
        import rasterio
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("rasterio is required to read GeoTIFF files") from exc

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        meta = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
            "dx": abs(src.transform.a),
            "dy": abs(src.transform.e),
        }
    return array, meta


def _binarize_mask(mask, threshold=0.5):
    mask = np.asarray(mask, dtype=np.float32)
    return (mask >= threshold).astype(np.float32)


def prepare_single_multisource_sample(
    dem_path,
    gravity_path,
    magnetic_path,
    mask_path,
    output_stack_path,
    output_mask_path,
    output_channel_names_path=None,
    scales=(1, 2, 4),
    multiscale_feature_keys=("dem", "gravity_residual", "magnetic_residual", "gravity_tilt", "magnetic_tilt"),
    regional_sigma=12.0,
    use_harmonica_derivatives=False,
    mask_threshold=0.5,
):
    """
    Prepare one training sample from aligned geophysical rasters.

    The output stack is stored as .npy with shape (H, W, C).
    The output mask is stored as .npy with shape (H, W).
    """
    dem, dem_meta = _read_single_band_raster(dem_path)
    gravity, gravity_meta = _read_single_band_raster(gravity_path)
    magnetic, magnetic_meta = _read_single_band_raster(magnetic_path)
    mask, mask_meta = _read_single_band_raster(mask_path)

    shape = dem.shape
    if gravity.shape != shape or magnetic.shape != shape or mask.shape != shape:
        raise ValueError(
            "All input rasters must have the same shape. "
            f"Got dem={dem.shape}, gravity={gravity.shape}, magnetic={magnetic.shape}, mask={mask.shape}"
        )

    # Use DEM spacing as reference.
    dx = dem_meta["dx"] if dem_meta["dx"] > 0 else 1.0
    dy = dem_meta["dy"] if dem_meta["dy"] > 0 else 1.0

    _, _, feature_dict = build_multisource_feature_stack(
        dem=dem,
        gravity=gravity,
        magnetic=magnetic,
        dx=dx,
        dy=dy,
        regional_sigma=regional_sigma,
        use_harmonica_derivatives=use_harmonica_derivatives,
    )

    features_with_multiscale = append_multiscale_channels(
        feature_dict,
        target_keys=multiscale_feature_keys,
        scales=scales,
    )

    channel_names = list(features_with_multiscale.keys())
    stack = np.stack([features_with_multiscale[name] for name in channel_names], axis=-1).astype(np.float32)
    mask_binary = _binarize_mask(mask, threshold=mask_threshold)

    output_stack_path = Path(output_stack_path)
    output_mask_path = Path(output_mask_path)
    output_stack_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_stack_path, stack)
    np.save(output_mask_path, mask_binary)

    if output_channel_names_path is not None:
        output_channel_names_path = Path(output_channel_names_path)
        output_channel_names_path.parent.mkdir(parents=True, exist_ok=True)
        with output_channel_names_path.open("w", encoding="utf-8") as f:
            json.dump(channel_names, f, indent=2)

    return {
        "stack_path": str(output_stack_path),
        "mask_path": str(output_mask_path),
        "channels": channel_names,
        "shape": stack.shape,
        "dx": dx,
        "dy": dy,
        "metadata": {
            "dem": dem_meta,
            "gravity": gravity_meta,
            "magnetic": magnetic_meta,
            "mask": mask_meta,
        },
    }


def _index_tif_by_stem(directory):
    directory = Path(directory)
    files = sorted(list(directory.glob("*.tif")) + list(directory.glob("*.tiff")))
    return {f.stem: f for f in files}


def prepare_multisource_dataset_from_directories(
    dem_dir,
    gravity_dir,
    magnetic_dir,
    mask_dir,
    output_stack_dir,
    output_mask_dir,
    output_channel_names_path=None,
    scales=(1, 2, 4),
    multiscale_feature_keys=("dem", "gravity_residual", "magnetic_residual", "gravity_tilt", "magnetic_tilt"),
    regional_sigma=12.0,
    use_harmonica_derivatives=False,
    mask_threshold=0.5,
):
    """
    Batch-prepare a multisource dataset using matching raster stems.

    Expected convention:
    dem_dir/scene_001.tif
    gravity_dir/scene_001.tif
    magnetic_dir/scene_001.tif
    mask_dir/scene_001.tif
    """
    dem_map = _index_tif_by_stem(dem_dir)
    gravity_map = _index_tif_by_stem(gravity_dir)
    magnetic_map = _index_tif_by_stem(magnetic_dir)
    mask_map = _index_tif_by_stem(mask_dir)

    common_stems = sorted(set(dem_map) & set(gravity_map) & set(magnetic_map) & set(mask_map))
    if not common_stems:
        raise ValueError("No common file stems found across DEM/Gravity/Magnetic/Mask directories")

    out_stack_dir = Path(output_stack_dir)
    out_mask_dir = Path(output_mask_dir)
    out_stack_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    dataset_report = []
    shared_channel_names = None

    for stem in common_stems:
        report = prepare_single_multisource_sample(
            dem_path=dem_map[stem],
            gravity_path=gravity_map[stem],
            magnetic_path=magnetic_map[stem],
            mask_path=mask_map[stem],
            output_stack_path=out_stack_dir / f"{stem}.npy",
            output_mask_path=out_mask_dir / f"{stem}.npy",
            output_channel_names_path=None,
            scales=scales,
            multiscale_feature_keys=multiscale_feature_keys,
            regional_sigma=regional_sigma,
            use_harmonica_derivatives=use_harmonica_derivatives,
            mask_threshold=mask_threshold,
        )

        if shared_channel_names is None:
            shared_channel_names = report["channels"]

        dataset_report.append({
            "stem": stem,
            "stack_path": report["stack_path"],
            "mask_path": report["mask_path"],
            "shape": report["shape"],
        })

    if output_channel_names_path is not None and shared_channel_names is not None:
        output_channel_names_path = Path(output_channel_names_path)
        output_channel_names_path.parent.mkdir(parents=True, exist_ok=True)
        with output_channel_names_path.open("w", encoding="utf-8") as f:
            json.dump(shared_channel_names, f, indent=2)

    return {
        "num_samples": len(dataset_report),
        "samples": dataset_report,
        "channel_names": shared_channel_names or [],
    }
