#!/usr/bin/env python
"""
Generate PDF reports from all existing experiment data on disk.

Produces:
  reports/atrato_baseline.pdf         – Atrato tile-set statistics + checkpoint info
  reports/colombia_scratch.pdf        – Scratch-trained Colombia model
  reports/colombia_finetune_v3.pdf    – Fine-tuned Colombia v3 + evaluation
  reports/colombia_v4.pdf             – Colombia v4 (40-epoch fine-tune)
  reports/multisource_inference.pdf   – Multisource uncertainty-inference results
  reports/full_experiment.pdf         – All figures bundled together
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from plotting.report_pdf import generate_training_report_pdf

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DATA_DIR      = ROOT / "data"
FIGURES_DIR   = ROOT / "figures"
TEST_DIR      = ROOT / "tmp_multisource_test"
EVAL_DIR      = ROOT / "tmp_evaluation_v3"
CHECKPOINTS   = DATA_DIR / "Colombia" / "checkpoints"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_tif(path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
        return arr
    except Exception:
        return None


def _clip_pct(arr, lo=2, hi=98):
    v0 = float(np.nanpercentile(arr, lo))
    v1 = float(np.nanpercentile(arr, hi))
    return np.clip(arr, v0, v1), v0, v1


def _load_png_as_figure(png_path, title=None):
    """Wrap an existing PNG in a matplotlib figure for PDF embedding."""
    img = plt.imread(str(png_path))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0)
    return fig


def _dataset_stats_figure(report):
    """Histogram of positive_fraction and nan_fraction from a tile report."""
    samples = report.get("samples", [])
    pos = [s["positive_fraction"] * 100 for s in samples]
    nan = [s["nan_fraction"] * 100 for s in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Dataset tile statistics", fontsize=12, fontweight="bold")

    axes[0].hist(pos, bins=30, color="#2E86AB", edgecolor="white", linewidth=0.5)
    axes[0].axvline(np.mean(pos), color="#E84855", linestyle="--", linewidth=1.5,
                    label=f"mean {np.mean(pos):.3f}%")
    axes[0].set_title("Positive-pixel fraction (%)")
    axes[0].set_xlabel("%")
    axes[0].set_ylabel("Tiles")
    axes[0].legend(fontsize=8)

    axes[1].hist(nan, bins=30, color="#A23B72", edgecolor="white", linewidth=0.5)
    axes[1].axvline(np.mean(nan), color="#E84855", linestyle="--", linewidth=1.5,
                    label=f"mean {np.mean(nan):.3f}%")
    axes[1].set_title("NaN fraction (%)")
    axes[1].set_xlabel("%")
    axes[1].set_ylabel("Tiles")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def _pr_curves_figure(eval_data):
    """Bar chart comparing precision / recall across optimal-threshold targets."""
    opt = eval_data.get("optimal_thresholds", {})
    if not opt:
        return None

    targets = list(opt.keys())
    precisions = [opt[t]["precision"] for t in targets]
    recalls    = [opt[t]["recall"]    for t in targets]
    f1s        = [opt[t]["f1"]        for t in targets]

    x = np.arange(len(targets))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w, precisions, w, label="Precision", color="#3498db")
    ax.bar(x,     recalls,    w, label="Recall",    color="#2ecc71")
    ax.bar(x + w, f1s,        w, label="F1",        color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall≥{t}" for t in targets], fontsize=8)
    ax.set_title("Precision / Recall / F1 at optimal thresholds", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _sample_stack_figure(stack_path, mask_path=None, channel_names=None, n_show=5, title=""):
    """Show the first n_show channels of a stack + optional mask."""
    stack = np.load(stack_path).astype(np.float32)   # (H, W, C)
    n_show = min(n_show, stack.shape[2])
    cols = n_show + (1 if mask_path is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 3.5))
    if cols == 1:
        axes = [axes]
    fig.suptitle(title or Path(stack_path).stem, fontsize=10)

    for i in range(n_show):
        ch, v0, v1 = _clip_pct(stack[:, :, i])
        name = channel_names[i] if channel_names and i < len(channel_names) else f"ch{i}"
        im = axes[i].imshow(ch, cmap="RdBu_r", vmin=v0, vmax=v1, aspect="auto")
        axes[i].set_title(name, fontsize=7)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if mask_path is not None:
        mask = np.load(mask_path).astype(np.float32)
        axes[-1].imshow(mask, cmap="Greys_r", vmin=0, vmax=1, aspect="auto")
        axes[-1].set_title("Mask", fontsize=7)
        axes[-1].axis("off")

    fig.tight_layout()
    return fig


def _uncertainty_figure(scene, base_dir, title=None):
    """Load rasterio uncertainty TIFs and build a figure."""
    keys = ["mean", "std", "entropy", "mutual_information"]
    fnames = {k: base_dir / f"{scene}_{k}.tif" for k in keys}
    arrs = {k: _read_tif(v) for k, v in fnames.items() if v.exists()}
    if not arrs:
        return None

    n = len(arrs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title or f"Uncertainty – {scene}", fontsize=11, fontweight="bold")

    cmaps = {"mean": "gray", "std": "hot", "entropy": "viridis", "mutual_information": "plasma"}
    for ax, (key, arr) in zip(axes, arrs.items()):
        arr_c, v0, v1 = _clip_pct(arr)
        im = ax.imshow(arr_c, cmap=cmaps.get(key, "viridis"), vmin=v0, vmax=v1)
        ax.set_title(key.replace("_", " ").title(), fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def _gt_vs_pred_figure(scene, dem_dir, mask_dir, pred_dir, title=None):
    """DEM / GT mask / predicted mask overlay figure."""
    dem  = _read_tif(dem_dir  / f"{scene}.tif")
    gt   = _read_tif(mask_dir / f"{scene}.tif")
    pred = _read_tif(pred_dir / f"{scene}_mask.tif")
    if dem is None or gt is None or pred is None:
        return None

    dem_c, vmin, vmax = _clip_pct(dem)
    gt_ov   = np.ma.masked_where(gt   < 0.5, gt)
    pred_ov = np.ma.masked_where(pred < 0.5, pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title or f"GT vs Prediction – {scene}", fontsize=11)

    axes[0].imshow(dem_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("DEM")
    axes[0].axis("off")

    axes[1].imshow(dem_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].imshow(gt_ov, cmap="autumn", alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth overlay")
    axes[1].axis("off")

    axes[2].imshow(dem_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[2].imshow(pred_ov, cmap="cool", alpha=0.85, vmin=0, vmax=1)
    axes[2].set_title("Prediction overlay")
    axes[2].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report 1 – Atrato baseline
# ---------------------------------------------------------------------------

def report_atrato():
    print("Generating: atrato_baseline.pdf …")
    report_data = _load_json(DATA_DIR / "atrato_training" / "report.json")

    samples = report_data.get("samples", [])
    metadata = {
        "Checkpoint":     "data/atrato_training/checkpoints/atrato_unet.pt",
        "Num tiles":      str(report_data.get("num_samples", len(samples))),
        "Tile size":      f"{report_data.get('tile_size', '?')} px",
        "Stride":         f"{report_data.get('stride', '?')} px",
        "Mean pos frac":  f"{np.mean([s['positive_fraction'] for s in samples]):.4f}",
        "Mean NaN frac":  f"{np.mean([s['nan_fraction']      for s in samples]):.4f}",
    }

    stats_fig = _dataset_stats_figure(report_data)

    # Load a few sample stacks
    extra_figs = [stats_fig]
    stack_dir = ROOT / "data" / "atrato_training" / "stacks"
    mask_dir  = ROOT / "data" / "atrato_training" / "masks"
    stack_files = sorted(stack_dir.glob("*.npy"))[:3]
    for sf in stack_files:
        mf = mask_dir / sf.name
        fig = _sample_stack_figure(sf, mf if mf.exists() else None,
                                   title=sf.stem)
        extra_figs.append(fig)

    path = generate_training_report_pdf(
        REPORTS_DIR / "atrato_baseline.pdf",
        title="Atrato Baseline — Dataset & Checkpoint Report",
        subtitle="Geophysical lineament detection · Atrato region",
        report_metadata=metadata,
        extra_figures=extra_figs,
    )
    for f in extra_figs:
        plt.close(f)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report 2 – Colombia scratch-trained model
# ---------------------------------------------------------------------------

def report_colombia_scratch():
    print("Generating: colombia_scratch.pdf …")
    hist = _load_json(CHECKPOINTS / "colombia_scratch_trained_history.json")
    test = _load_json(CHECKPOINTS / "colombia_scratch_trained_test_results.json")

    train_l = hist["train_losses"]
    val_l   = hist["val_losses"]

    summary = test.get("summary", {})
    metrics = {
        "Best val loss":     f"{hist['best_loss']:.6f}",
        "Train tiles":       str(hist.get("train_size", "?")),
        "Val tiles":         str(hist.get("val_size",   "?")),
        "Test tiles":        str(hist.get("test_size",  "?")),
        "Avg Precision":     f"{summary.get('avg_precision', 0):.4f}",
        "Avg Recall":        f"{summary.get('avg_recall',    0):.4f}",
        "Avg F1":            f"{summary.get('avg_f1',        0):.4f}",
        "Avg IoU":           f"{summary.get('avg_iou',       0):.4f}",
    }

    # Per-tile bar chart
    tile_metrics = test.get("test_tiles", {})
    extra_figs = []
    if tile_metrics:
        names = list(tile_metrics.keys())
        short = [n.split("_r")[1] if "_r" in n else n[:12] for n in names]
        f1s = [tile_metrics[n]["f1"] for n in names]
        ious = [tile_metrics[n]["iou"] for n in names]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4))
        x = np.arange(len(names))
        ax.bar(x - 0.2, f1s,  0.35, label="F1",  color="#3498db")
        ax.bar(x + 0.2, ious, 0.35, label="IoU", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_title("Per-tile F1 and IoU (test set)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        extra_figs.append(fig)

    path = generate_training_report_pdf(
        REPORTS_DIR / "colombia_scratch.pdf",
        title="Colombia — Scratch-Trained U-Net",
        subtitle="5-channel geophysical stack · Scratch training from random init",
        report_metadata={"Checkpoint": "colombia_scratch_trained.pt",
                         "Input channels": str(hist.get("input_channels", 5)),
                         "Epochs": str(len(train_l))},
        train_losses=train_l,
        val_losses=val_l,
        metrics_dict=metrics,
        extra_figures=extra_figs,
    )
    for f in extra_figs:
        plt.close(f)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report 3 – Colombia fine-tune v3 + evaluation
# ---------------------------------------------------------------------------

def report_colombia_v3():
    print("Generating: colombia_finetune_v3.pdf …")
    hist = _load_json(CHECKPOINTS / "colombia_finetune_history_v3.json")
    evl  = _load_json(CHECKPOINTS / "colombia_finetuned_unet_v3_eval.json")

    train_l = hist["train_losses"]
    val_l   = hist["val_losses"]
    loss_cfg = hist.get("loss_config", {})

    baseline = evl.get("baseline_metrics_0_5", {})
    opt      = evl.get("optimal_thresholds",   {})
    best_opt = opt.get("0.7", {})

    metrics = {
        "Best val loss":        f"{hist['best_loss']:.6f}",
        "Input channels":       str(hist.get("input_channels", 5)),
        "Fine-tuned from":      str(hist.get("init_checkpoint", "?")),
        "Loss mode":            str(loss_cfg.get("mode", "?")),
        "BCE weight":           str(loss_cfg.get("bce_weight", "?")),
        "Pos weight":           str(loss_cfg.get("pos_weight", "?")),
        "ECE (calibration)":    f"{evl['calibration_error']['ece']:.4f}",
        "GT positive fraction": f"{evl['dataset']['gt_positive_fraction']:.5f}",
        "N tiles (eval)":       str(evl["dataset"]["n_tiles"]),
        "Baseline Acc @0.5":    f"{baseline.get('accuracy', 0):.4f}",
        "Opt Precision @0.7":   f"{best_opt.get('precision', 0):.4f}",
        "Opt Recall @0.7":      f"{best_opt.get('recall', 0):.4f}",
        "Opt F1 @0.7":          f"{best_opt.get('f1', 0):.4f}",
        "Opt IoU @0.7":         f"{best_opt.get('iou', 0):.4f}",
    }

    pr_fig = _pr_curves_figure(evl)
    extra_figs = [pr_fig] if pr_fig else []

    # Embed existing evaluation PNGs
    for png in ["calibration_curve.png", "pr_curves.png"]:
        p = EVAL_DIR / png
        if p.exists():
            extra_figs.append(_load_png_as_figure(p, title=png.replace("_", " ").replace(".png", "")))

    path = generate_training_report_pdf(
        REPORTS_DIR / "colombia_finetune_v3.pdf",
        title="Colombia Fine-tune v3 — Evaluation Report",
        subtitle="U-Net fine-tuned from Atrato baseline · BCE + Focal loss",
        report_metadata={"Checkpoint": "colombia_finetuned_unet_v3.pt",
                         "Fine-tune epochs": str(len(train_l)),
                         "Split": f"{evl['finetune_split']['n_train']} train / {evl['finetune_split']['n_val']} val"},
        train_losses=train_l,
        val_losses=val_l,
        metrics_dict=metrics,
        extra_figures=extra_figs,
    )
    for f in extra_figs:
        plt.close(f)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report 4 – Colombia v4 (40-epoch focal loss)
# ---------------------------------------------------------------------------

def report_colombia_v4():
    print("Generating: colombia_v4.pdf …")
    hist = _load_json(CHECKPOINTS / "history_v4.json")

    train_l = hist["train_losses"]
    val_l   = hist["val_losses"]
    loss_cfg = hist.get("loss_config", {})

    metrics = {
        "Best val loss":   f"{hist['best_loss']:.6f}",
        "Epochs":          str(len(train_l)),
        "Input channels":  str(hist.get("input_channels", 5)),
        "Pre-trained from": str(hist.get("init_checkpoint", "?")),
        "Loss mode":       str(loss_cfg.get("mode", "?")),
        "Focal alpha":     str(loss_cfg.get("focal_alpha", "?")),
        "Focal gamma":     str(loss_cfg.get("focal_gamma", "?")),
        "BCE weight":      str(loss_cfg.get("bce_weight", "?")),
    }

    path = generate_training_report_pdf(
        REPORTS_DIR / "colombia_v4.pdf",
        title="Colombia v4 — 40-Epoch Focal-Loss Fine-tune",
        subtitle="U-Net fine-tuned from Atrato baseline · Focal loss (α=0.75, γ=2)",
        report_metadata={"Checkpoint": "colombia_v4.pt"},
        train_losses=train_l,
        val_losses=val_l,
        metrics_dict=metrics,
    )
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report 5 – Multisource inference (scene_001 & scene_002)
# ---------------------------------------------------------------------------

def report_multisource_inference():
    print("Generating: multisource_inference.pdf …")

    channels_full = json.loads((TEST_DIR / "channels_full.json").read_text(encoding="utf-8"))
    scenes = ["scene_001", "scene_002"]

    extra_figs = []

    for scene in scenes:
        stack_path = TEST_DIR / "out_stacks_full" / f"{scene}.npy"
        mask_path  = TEST_DIR / "out_masks_full"  / f"{scene}.npy"

        if stack_path.exists():
            fig = _sample_stack_figure(
                stack_path,
                mask_path if mask_path.exists() else None,
                channel_names=channels_full,
                n_show=5,
                title=f"Feature channels – {scene}",
            )
            extra_figs.append(fig)

        # GT vs prediction
        fig_pred = _gt_vs_pred_figure(
            scene,
            dem_dir  = TEST_DIR / "dem",
            mask_dir = TEST_DIR / "mask",
            pred_dir = TEST_DIR / "uncertainty_full",
            title    = f"GT vs Prediction – {scene}",
        )
        if fig_pred:
            extra_figs.append(fig_pred)

        # Uncertainty maps
        fig_unc = _uncertainty_figure(scene, TEST_DIR / "uncertainty_full",
                                      title=f"Uncertainty maps – {scene}")
        if fig_unc:
            extra_figs.append(fig_unc)

    path = generate_training_report_pdf(
        REPORTS_DIR / "multisource_inference.pdf",
        title="Multisource Inference — Uncertainty Maps Report",
        subtitle="MC-Dropout uncertainty on scene_001 and scene_002",
        report_metadata={
            "Model":      "unet26_ckpt.pt",
            "Channels":   str(len(channels_full)),
            "MC samples": "30",
            "Scenes":     ", ".join(scenes),
        },
        extra_figures=extra_figs,
    )
    for f in extra_figs:
        plt.close(f)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Report 6 – Full experiment (all existing figures)
# ---------------------------------------------------------------------------

def report_full_experiment():
    print("Generating: full_experiment.pdf …")

    existing_pngs = sorted(FIGURES_DIR.glob("*.png"))
    extra_figs = [_load_png_as_figure(p, title=p.stem.replace("_", " ")) for p in existing_pngs]

    # Append all per-version loss curves together
    histories = {
        "Scratch":     CHECKPOINTS / "colombia_scratch_trained_history.json",
        "Finetune v3": CHECKPOINTS / "colombia_finetune_history_v3.json",
        "v4 (40ep)":   CHECKPOINTS / "history_v4.json",
    }
    fig_all, ax_all = plt.subplots(figsize=(11, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for (label, path), color in zip(histories.items(), colors):
        h = _load_json(path)
        tl = h["train_losses"]
        vl = h["val_losses"]
        ep = range(1, len(tl) + 1)
        ax_all.plot(ep, tl, color=color, linestyle="-",  label=f"{label} train")
        ax_all.plot(ep, vl, color=color, linestyle="--", label=f"{label} val")
    ax_all.set_title("All model training curves")
    ax_all.set_xlabel("Epoch")
    ax_all.set_ylabel("Loss")
    ax_all.legend(fontsize=8)
    ax_all.grid(alpha=0.3)
    fig_all.tight_layout()
    extra_figs.append(fig_all)

    path = generate_training_report_pdf(
        REPORTS_DIR / "full_experiment.pdf",
        title="Geological Lineament Detection — Full Experiment Report",
        subtitle="Colombia · Atrato region · Multisource geophysical data",
        report_metadata={
            "Institution": "Universidad EAFIT",
            "Funding":     "Minciencias",
            "Date":        "2025–2026",
        },
        extra_figures=extra_figs,
    )
    for f in extra_figs:
        plt.close(f)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report_atrato()
    report_colombia_scratch()
    report_colombia_v3()
    report_colombia_v4()
    report_multisource_inference()
    report_full_experiment()

    print(f"\nAll reports saved to: {REPORTS_DIR}")
