#!/usr/bin/env python
"""Generate presentation figures (PNG) for lineament detection results."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "tmp_multisource_test"
DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "figures"


def _read_tif(path):
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _clip_pct(arr, lo=2.0, hi=98.0):
    vmin = float(np.nanpercentile(arr, lo))
    vmax = float(np.nanpercentile(arr, hi))
    return np.clip(arr, vmin, vmax), vmin, vmax


def _show(ax, arr, cmap, title, vmin=None, vmax=None, cbar=True):
    clipped, auto_min, auto_max = _clip_pct(arr)
    im = ax.imshow(
        clipped,
        cmap=cmap,
        vmin=auto_min if vmin is None else vmin,
        vmax=auto_max if vmax is None else vmax,
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def fig2_inputs(scene, out_dir):
    dem = _read_tif(TEST_DIR / "dem" / f"{scene}.tif")
    grav = _read_tif(TEST_DIR / "grav" / f"{scene}.tif")
    mag = _read_tif(TEST_DIR / "mag" / f"{scene}.tif")
    mask = _read_tif(TEST_DIR / "mask" / f"{scene}.tif")

    stack = np.load(TEST_DIR / "out_stacks_full" / f"{scene}.npy").astype(np.float32)
    channels = json.loads((TEST_DIR / "channels_full.json").read_text(encoding="utf-8"))
    idx = {name: i for i, name in enumerate(channels)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Figura 2 - Entradas por escena ({scene})", fontsize=13)

    _show(axes[0, 0], dem, "terrain", "DEM")
    _show(axes[0, 1], grav, "RdBu_r", "Gravedad")
    _show(axes[0, 2], mag, "RdBu_r", "Magnetica")
    _show(axes[1, 0], stack[:, :, idx["gravity_residual"]], "RdBu_r", "Gravedad residual")
    _show(axes[1, 1], stack[:, :, idx["magnetic_residual"]], "RdBu_r", "Magnetica residual")
    _show(axes[1, 2], mask, "Greys_r", "Mascara de referencia", vmin=0, vmax=1, cbar=False)

    fig.tight_layout()
    out = out_dir / "fig2_input_mosaic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_channels(scene, out_dir):
    stack = np.load(TEST_DIR / "out_stacks_full" / f"{scene}.npy").astype(np.float32)
    channels = json.loads((TEST_DIR / "channels_full.json").read_text(encoding="utf-8"))
    idx = {name: i for i, name in enumerate(channels)}

    specs = [
        ("dem_slope_deg", "terrain", "Pendiente DEM"),
        ("dem_curvature", "RdBu_r", "Curvatura DEM"),
        ("gravity_tga", "hot_r", "Gravedad TGA"),
        ("gravity_tilt", "RdBu_r", "Gravedad tilt"),
        ("magnetic_tga", "hot_r", "Magnetica TGA"),
        ("magnetic_tilt", "RdBu_r", "Magnetica tilt"),
        ("gravity_residual_x1", "RdBu_r", "Grav residual x1"),
        ("magnetic_residual_x1", "RdBu_r", "Mag residual x1"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Figura 3 - Canales derivados ({scene})", fontsize=13)

    for ax, (name, cmap, title) in zip(axes.flatten(), specs):
        _show(ax, stack[:, :, idx[name]], cmap, title)

    fig.tight_layout()
    out = out_dir / "fig3_derived_channels.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_gt_vs_pred(scene, out_dir):
    dem = _read_tif(TEST_DIR / "dem" / f"{scene}.tif")
    gt = _read_tif(TEST_DIR / "mask" / f"{scene}.tif")
    pred = _read_tif(TEST_DIR / "uncertainty_full" / f"{scene}_mask.tif")

    dem_c, vmin, vmax = _clip_pct(dem)
    gt_overlay = np.ma.masked_where(gt < 0.5, gt)
    pred_overlay = np.ma.masked_where(pred < 0.5, pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Figura 4 - Ground truth vs prediccion ({scene})", fontsize=13)

    _show(axes[0], dem, "terrain", "DEM", cbar=False)
    axes[1].imshow(dem_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].imshow(gt_overlay, cmap="autumn", alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(dem_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[2].imshow(pred_overlay, cmap="cool", alpha=0.85, vmin=0, vmax=1)
    axes[2].set_title("Prediccion")
    axes[2].axis("off")

    fig.tight_layout()
    out = out_dir / "fig4_gt_vs_pred.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_uncertainty(scene, out_dir):
    mean = _read_tif(TEST_DIR / "uncertainty_full" / f"{scene}_mean.tif")
    std = _read_tif(TEST_DIR / "uncertainty_full" / f"{scene}_std.tif")
    mi = _read_tif(TEST_DIR / "uncertainty_full" / f"{scene}_mutual_information.tif")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Figura 5 - Prediccion e incertidumbre ({scene})", fontsize=13)

    _show(axes[0], mean, "hot_r", "Mean")
    _show(axes[1], std, "viridis", "Std")
    _show(axes[2], mi, "plasma", "Mutual information")

    fig.tight_layout()
    out = out_dir / "fig5_uncertainty_maps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6_training_stats(out_dir):
    report_path = DATA_DIR / "atrato_training" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = report.get("samples", [])

    pos = [row["positive_fraction"] * 100.0 for row in samples]
    nan = [row["nan_fraction"] * 100.0 for row in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figura 6 - Distribuciones de entrenamiento", fontsize=13)

    axes[0].hist(pos, bins=30, color="#2E86AB", edgecolor="white", linewidth=0.5)
    axes[0].axvline(np.mean(pos), color="#E84855", linestyle="--", linewidth=1.5)
    axes[0].set_title("positive_fraction (%)")
    axes[0].set_xlabel("%")
    axes[0].set_ylabel("Tiles")

    axes[1].hist(nan, bins=30, color="#A23B72", edgecolor="white", linewidth=0.5)
    axes[1].axvline(np.mean(nan), color="#E84855", linestyle="--", linewidth=1.5)
    axes[1].set_title("nan_fraction (%)")
    axes[1].set_xlabel("%")
    axes[1].set_ylabel("Tiles")

    fig.tight_layout()
    out = out_dir / "fig6_training_stats.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_curves(out_dir, history_path=None):
    if history_path is None:
        candidates = sorted((DATA_DIR / "Colombia" / "checkpoints").glob("*history*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        history_path = candidates[0]

    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, "o-", label="train")
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, "s--", label="val")

    ax.set_title("Figura 7 - Curvas de entrenamiento")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = out_dir / "fig7_training_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate presentation figures for the repository")
    parser.add_argument("--scene", default="scene_001")
    parser.add_argument("--history", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["fig2"] = fig2_inputs(args.scene, out_dir)
    results["fig3"] = fig3_channels(args.scene, out_dir)
    results["fig4"] = fig4_gt_vs_pred(args.scene, out_dir)
    results["fig5"] = fig5_uncertainty(args.scene, out_dir)
    results["fig6"] = fig6_training_stats(out_dir)
    results["fig7"] = fig7_curves(out_dir, args.history)

    print("Summary:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
