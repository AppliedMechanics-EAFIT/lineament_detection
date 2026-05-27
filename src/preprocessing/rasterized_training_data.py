import json
from pathlib import Path

import numpy as np


def _iter_tile_offsets(size, tile_size, stride):
    if size < tile_size:
        return []
    offsets = list(range(0, size - tile_size + 1, stride))
    last_offset = size - tile_size
    if not offsets or offsets[-1] != last_offset:
        offsets.append(last_offset)
    return offsets


def rasterize_line_vectors_to_mask(
    reference_raster_path,
    vector_paths,
    output_mask_raster_path=None,
    attribute_filter=None,
    burn_value=1.0,
    fill_value=0.0,
    all_touched=True,
):
    """
    Rasterize line vectors onto the grid of a reference raster.

    Parameters
    ----------
    reference_raster_path : str or Path
        Multiband or single-band raster that defines target transform, shape and CRS.
    vector_paths : sequence[str | Path]
        One or more vector files containing lineaments.
    attribute_filter : tuple[str, object] | None
        Optional (column, value) filter applied after loading each vector.
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("geopandas and rasterio are required to rasterize vector labels") from exc

    with rasterio.open(reference_raster_path) as src:
        shape = src.shape
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    geometries = []
    for vector_path in vector_paths:
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            continue
        if attribute_filter is not None:
            column, value = attribute_filter
            if column in gdf.columns:
                gdf = gdf[gdf[column] == value]
        gdf = gdf[gdf.geometry.notnull()]
        if gdf.empty:
            continue
        if crs is not None and gdf.crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        geometries.extend(geom for geom in gdf.geometry if geom is not None and not geom.is_empty)

    if not geometries:
        raise ValueError("No valid vector geometries were found for rasterization")

    mask = rasterize(
        [(geom, burn_value) for geom in geometries],
        out_shape=shape,
        transform=transform,
        fill=fill_value,
        dtype="float32",
        all_touched=all_touched,
    )

    if output_mask_raster_path is not None:
        output_mask_raster_path = Path(output_mask_raster_path)
        output_mask_raster_path.parent.mkdir(parents=True, exist_ok=True)
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="float32", nodata=fill_value, compress="lzw")
        with rasterio.open(output_mask_raster_path, "w", **out_profile) as dst:
            dst.write(mask.astype(np.float32), 1)

    return mask.astype(np.float32)


def tile_prestacked_raster_and_mask(
    stack_raster_path,
    mask_raster_path,
    output_stack_dir,
    output_mask_dir,
    tile_size=512,
    stride=None,
    min_positive_fraction=0.0,
    max_nan_fraction=0.25,
    drop_all_zero_tiles=False,
    output_report_path=None,
):
    """
    Convert a large pre-stacked raster plus aligned mask raster into .npy tiles.

    Stack tiles are saved as (H, W, C); mask tiles are saved as (H, W).
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("rasterio is required to tile prestacked rasters") from exc

    stride = tile_size if stride is None else stride
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive integers")

    output_stack_dir = Path(output_stack_dir)
    output_mask_dir = Path(output_mask_dir)
    output_stack_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    with rasterio.open(stack_raster_path) as stack_src, rasterio.open(mask_raster_path) as mask_src:
        if stack_src.shape != mask_src.shape:
            raise ValueError(
                f"Stack and mask rasters must have the same shape, got {stack_src.shape} and {mask_src.shape}"
            )

        height, width = stack_src.height, stack_src.width
        stem = Path(stack_raster_path).stem

        row_offsets = _iter_tile_offsets(height, tile_size, stride)
        col_offsets = _iter_tile_offsets(width, tile_size, stride)
        for row in row_offsets:
            for col in col_offsets:
                window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)

                stack = stack_src.read(window=window).astype(np.float32)
                mask = mask_src.read(1, window=window).astype(np.float32)

                if stack.shape[1] != tile_size or stack.shape[2] != tile_size:
                    continue

                nan_fraction = float(np.isnan(stack).mean())
                if nan_fraction > max_nan_fraction:
                    continue

                stack = np.moveaxis(np.nan_to_num(stack, nan=0.0), 0, -1)
                mask = np.nan_to_num(mask, nan=0.0)
                positive_fraction = float((mask > 0.5).mean())

                if positive_fraction < min_positive_fraction:
                    continue
                if drop_all_zero_tiles and positive_fraction == 0.0:
                    continue

                sample_name = f"{stem}_r{row:05d}_c{col:05d}"
                stack_path = output_stack_dir / f"{sample_name}.npy"
                mask_path = output_mask_dir / f"{sample_name}.npy"
                np.save(stack_path, stack.astype(np.float32))
                np.save(mask_path, (mask > 0.5).astype(np.float32))

                samples.append({
                    "sample": sample_name,
                    "row": int(row),
                    "col": int(col),
                    "positive_fraction": positive_fraction,
                    "nan_fraction": nan_fraction,
                    "stack_path": str(stack_path),
                    "mask_path": str(mask_path),
                })

    report = {
        "num_samples": len(samples),
        "tile_size": tile_size,
        "stride": stride,
        "samples": samples,
    }

    if output_report_path is not None:
        output_report_path = Path(output_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with output_report_path.open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, indent=2)

    return report