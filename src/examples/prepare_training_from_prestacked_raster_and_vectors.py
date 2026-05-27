import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.preprocessing.rasterized_training_data import (
        rasterize_line_vectors_to_mask,
        tile_prestacked_raster_and_mask,
    )
except ModuleNotFoundError:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from preprocessing.rasterized_training_data import (
        rasterize_line_vectors_to_mask,
        tile_prestacked_raster_and_mask,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize line vectors and extract .npy training tiles from a pre-stacked raster"
    )
    parser.add_argument("--stack-raster", required=True)
    parser.add_argument("--vector", required=True, action="append", dest="vectors")
    parser.add_argument("--mask-raster-out", required=True)
    parser.add_argument("--stack-out-dir", required=True)
    parser.add_argument("--mask-out-dir", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-positive-fraction", type=float, default=0.001)
    parser.add_argument("--max-nan-fraction", type=float, default=0.25)
    parser.add_argument("--drop-all-zero-tiles", action="store_true")
    args = parser.parse_args()

    rasterize_line_vectors_to_mask(
        reference_raster_path=args.stack_raster,
        vector_paths=args.vectors,
        output_mask_raster_path=args.mask_raster_out,
    )

    report = tile_prestacked_raster_and_mask(
        stack_raster_path=args.stack_raster,
        mask_raster_path=args.mask_raster_out,
        output_stack_dir=args.stack_out_dir,
        output_mask_dir=args.mask_out_dir,
        tile_size=args.tile_size,
        stride=args.stride,
        min_positive_fraction=args.min_positive_fraction,
        max_nan_fraction=args.max_nan_fraction,
        drop_all_zero_tiles=args.drop_all_zero_tiles,
        output_report_path=args.report_out,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()