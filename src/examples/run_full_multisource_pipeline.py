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


def _resolve_reference_raster(dem_dir, stem):
    dem_dir = Path(dem_dir)
    tif = dem_dir / f"{stem}.tif"
    tiff = dem_dir / f"{stem}.tiff"
    if tif.exists():
        return tif
    if tiff.exists():
        return tiff
    raise FileNotFoundError(f"Reference raster for stem '{stem}' not found in {dem_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Full multisource workflow: prepare dataset + infer with MC Dropout + export GeoTIFFs"
    )

    # Input raster dirs
    parser.add_argument("--dem-dir", required=True)
    parser.add_argument("--gravity-dir", required=True)
    parser.add_argument("--magnetic-dir", required=True)
    parser.add_argument("--mask-dir", required=True)

    # Preparation outputs
    parser.add_argument("--out-stack-dir", required=True)
    parser.add_argument("--out-mask-dir", required=True)
    parser.add_argument("--out-channels", required=True)

    # Inference setup
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--uncertainty-out-dir", required=True)
    parser.add_argument("--arch", default="unet", choices=["unet"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "softmax", "none"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-mask", action="store_true")

    # Feature engineering config
    parser.add_argument("--regional-sigma", type=float, default=12.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])

    # Optional subset run
    parser.add_argument("--only-stems", nargs="+", default=None, help="Optional subset of stems to infer")

    args = parser.parse_args()

    import torch
    import rasterio

    from src.preprocessing.prepare_multisource_data import prepare_multisource_dataset_from_directories
    from src.training.multisource_pipeline import predict_stack_with_uncertainty
    from src.training.uncertainty_geotiff import export_uncertainty_maps_to_geotiff

    prep_report = prepare_multisource_dataset_from_directories(
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

    if prep_report["num_samples"] < 1:
        raise ValueError("No samples were prepared")

    stack_dir = Path(args.out_stack_dir)
    stack_files = sorted(stack_dir.glob("*.npy"))

    if args.only_stems:
        stems_filter = set(args.only_stems)
        stack_files = [p for p in stack_files if p.stem in stems_filter]

    if not stack_files:
        raise ValueError("No stacks available for inference after filtering")

    first_stack = np.load(stack_files[0]).astype(np.float32)
    in_channels = first_stack.shape[-1]

    model = _build_model(args.arch, in_channels=in_channels, out_channels=1)
    _load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device)

    uncertainty_out_dir = Path(args.uncertainty_out_dir)
    uncertainty_out_dir.mkdir(parents=True, exist_ok=True)

    inference_report = []

    for stack_path in stack_files:
        stem = stack_path.stem
        stack = np.load(stack_path).astype(np.float32)
        reference_raster = _resolve_reference_raster(args.dem_dir, stem)

        exported = export_uncertainty_maps_to_geotiff(
            model=model,
            stack_hwc=stack,
            reference_raster_path=reference_raster,
            output_dir=uncertainty_out_dir,
            prefix=stem,
            n_samples=args.n_samples,
            activation=args.activation,
            device=args.device,
        )

        if args.save_mask:
            pred = predict_stack_with_uncertainty(
                model=model,
                stack_hwc=stack,
                n_samples=args.n_samples,
                activation=args.activation,
                device=args.device,
            )
            mask = (pred["mean"] >= args.threshold).astype(np.float32)

            with rasterio.open(reference_raster) as src:
                profile = src.profile.copy()
            profile.update(dtype="float32", count=1, compress="lzw")

            mask_path = uncertainty_out_dir / f"{stem}_mask.tif"
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask, 1)
            exported["mask"] = str(mask_path)

        inference_report.append({
            "stem": stem,
            "stack_path": str(stack_path),
            "outputs": exported,
        })

    final_report = {
        "prepared_samples": prep_report["num_samples"],
        "prepared_channels": len(prep_report["channel_names"]),
        "inferred_samples": len(inference_report),
        "items": inference_report,
    }

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
