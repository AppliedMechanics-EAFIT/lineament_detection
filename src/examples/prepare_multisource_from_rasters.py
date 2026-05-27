import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.preprocessing.prepare_multisource_data import prepare_multisource_dataset_from_directories
except ModuleNotFoundError:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from preprocessing.prepare_multisource_data import prepare_multisource_dataset_from_directories


def main():
    parser = argparse.ArgumentParser(description="Prepare multisource .npy stacks from raster directories")
    parser.add_argument("--dem-dir", required=True)
    parser.add_argument("--gravity-dir", required=True)
    parser.add_argument("--magnetic-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out-stack-dir", required=True)
    parser.add_argument("--out-mask-dir", required=True)
    parser.add_argument("--out-channels", default=None)
    parser.add_argument("--regional-sigma", type=float, default=12.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    report = prepare_multisource_dataset_from_directories(
        dem_dir=args.dem_dir,
        gravity_dir=args.gravity_dir,
        magnetic_dir=args.magnetic_dir,
        mask_dir=args.mask_dir,
        output_stack_dir=args.out_stack_dir,
        output_mask_dir=args.out_mask_dir,
        output_channel_names_path=args.out_channels,
        scales=tuple(args.scales),
        regional_sigma=args.regional_sigma,
        mask_threshold=args.mask_threshold,
    )

    print(json.dumps({
        "num_samples": report["num_samples"],
        "num_channels": len(report["channel_names"]),
    }, indent=2))


if __name__ == "__main__":
    main()
