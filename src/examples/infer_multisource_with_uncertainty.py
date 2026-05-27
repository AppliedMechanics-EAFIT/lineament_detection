import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_stack(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected stack shape (H, W, C), got {arr.shape}")
    return arr


def _build_model(arch, in_channels, out_channels=1):
    if arch == "unet":
        from src.model.Org_UNet import UNet

        model = UNet(n_classes=out_channels, input_channels=in_channels)
        return model

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
            # Assume plain state_dict
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict or dict with state_dict/model_state_dict")


def main():
    parser = argparse.ArgumentParser(description="Inference with MC Dropout uncertainty + GeoTIFF export")
    parser.add_argument("--stack", required=True, help="Path to input stack .npy (H, W, C)")
    parser.add_argument("--reference-raster", required=True, help="Path to reference raster for CRS/transform")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt/.pth)")
    parser.add_argument("--output-dir", required=True, help="Output folder for GeoTIFF uncertainty maps")
    parser.add_argument("--prefix", default="prediction")
    parser.add_argument("--arch", default="unet", choices=["unet"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "softmax", "none"])
    parser.add_argument("--threshold", type=float, default=0.015, help="Threshold for binary segmentation from mean map (calibrated for 0.32%% positive class)")
    parser.add_argument("--save-mask", action="store_true", help="Also export thresholded prediction mask")
    args = parser.parse_args()

    import rasterio

    from src.training.multisource_pipeline import predict_stack_with_uncertainty
    from src.training.uncertainty_geotiff import export_uncertainty_maps_to_geotiff

    stack = _load_stack(args.stack)
    in_channels = stack.shape[-1]

    model = _build_model(args.arch, in_channels=in_channels, out_channels=1)
    _load_checkpoint(model, args.checkpoint, device=args.device)

    saved = export_uncertainty_maps_to_geotiff(
        model=model,
        stack_hwc=stack,
        reference_raster_path=args.reference_raster,
        output_dir=args.output_dir,
        prefix=args.prefix,
        n_samples=args.n_samples,
        activation=args.activation,
        device=args.device,
    )

    if args.save_mask:
        out = predict_stack_with_uncertainty(
            model=model,
            stack_hwc=stack,
            n_samples=args.n_samples,
            activation=args.activation,
            device=args.device,
        )
        mean_map = out["mean"]
        mask = (mean_map >= args.threshold).astype(np.float32)

        with rasterio.open(args.reference_raster) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw")

        mask_path = Path(args.output_dir) / f"{args.prefix}_mask.tif"
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask, 1)
        saved["mask"] = str(mask_path)

    print(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
