import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_model(arch, in_channels, out_channels=1):
    if arch == "unet":
        from src.model.Org_UNet import UNet

        return UNet(n_classes=out_channels, input_channels=in_channels)

    raise ValueError(f"Unsupported architecture '{arch}'. Currently supported: unet")


def _load_checkpoint(model, checkpoint_path, device):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format. Expected dict or state_dict-like object")


def _normalize_stack_hwc(stack, eps=1e-6):
    mean = stack.mean(axis=(0, 1), keepdims=True)
    std = stack.std(axis=(0, 1), keepdims=True)
    return (stack - mean) / (std + eps)


def main():
    parser = argparse.ArgumentParser(
        description="Tile-based inference with MC Dropout on a pre-stacked multiband raster"
    )
    parser.add_argument("--stack-raster", required=True, help="Multiband raster with shape bands x H x W")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="prediction")
    parser.add_argument("--arch", default="unet", choices=["unet"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "softmax", "none"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Apply per-tile z-score normalization")
    args = parser.parse_args()

    import rasterio
    from rasterio.windows import Window
    from src.training.multisource_pipeline import predict_stack_with_uncertainty

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.stack_raster) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        in_channels = src.count

        model = _build_model(args.arch, in_channels=in_channels, out_channels=1)
        _load_checkpoint(model, args.checkpoint, device=args.device)
        model = model.to(args.device)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

        keys = ["mean", "std", "entropy", "mutual_information"]
        out_paths = {key: output_dir / f"{args.prefix}_{key}.tif" for key in keys}
        if args.save_mask:
            out_paths["mask"] = output_dir / f"{args.prefix}_mask.tif"

        outputs = {key: rasterio.open(path, "w", **out_profile) for key, path in out_paths.items()}

        try:
            for row in range(0, height, args.tile_size):
                tile_h = min(args.tile_size, height - row)
                for col in range(0, width, args.tile_size):
                    tile_w = min(args.tile_size, width - col)
                    window = Window(col_off=col, row_off=row, width=tile_w, height=tile_h)

                    tile = src.read(window=window).astype(np.float32)  # (C, H, W)
                    tile = np.moveaxis(tile, 0, -1)  # (H, W, C)

                    # Replace NaNs by nearest safe value for model input.
                    tile = np.nan_to_num(tile, nan=0.0)
                    if args.normalize:
                        tile = _normalize_stack_hwc(tile)

                    pred = predict_stack_with_uncertainty(
                        model=model,
                        stack_hwc=tile,
                        n_samples=args.n_samples,
                        activation=args.activation,
                        device=args.device,
                    )

                    for key in keys:
                        outputs[key].write(pred[key].astype(np.float32), 1, window=window)

                    if args.save_mask:
                        mask = (pred["mean"] >= args.threshold).astype(np.float32)
                        outputs["mask"].write(mask, 1, window=window)
        finally:
            for dataset in outputs.values():
                dataset.close()

    print(json.dumps({key: str(path) for key, path in out_paths.items()}, indent=2))


if __name__ == "__main__":
    main()
