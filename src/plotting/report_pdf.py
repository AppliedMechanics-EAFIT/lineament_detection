"""
PDF report generation for lineament detection experiments.

Uses only matplotlib (already a core dependency) via PdfPages — no extra
packages required.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tensor_to_numpy(x):
    """Convert a torch Tensor or numpy array to a plottable 2-D/3-D numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    return x


def _add_title_page(pdf, title, subtitle=None, metadata=None):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.65, title, color="white", fontsize=26, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes, wrap=True)
    if subtitle:
        ax.text(0.5, 0.50, subtitle, color="#a0a0c0", fontsize=14,
                ha="center", va="center", transform=ax.transAxes)
    if metadata:
        lines = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        ax.text(0.5, 0.30, lines, color="#808090", fontsize=10,
                ha="center", va="center", transform=ax.transAxes,
                family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _add_loss_curves_page(pdf, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", markersize=4, label="Train loss")
    ax.plot(epochs, val_losses,   "r-o", markersize=4, label="Val loss")
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val   = min(val_losses)
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
               label=f"Best val epoch {best_epoch} ({best_val:.4f})")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_prediction_grid_page(pdf, samples, page_title="Prediction Samples"):
    """
    samples : list of dicts with keys 'image', 'label', 'prediction'
              and optionally 'title'.
    Fits up to 4 rows per page.
    """
    n = len(samples)
    if n == 0:
        return

    chunk_size = 4
    for chunk_start in range(0, n, chunk_size):
        chunk = samples[chunk_start:chunk_start + chunk_size]
        rows = len(chunk)
        fig = plt.figure(figsize=(15, 4 * rows))
        fig.suptitle(page_title, fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(rows, 3, figure=fig, hspace=0.35, wspace=0.1)

        for r, sample in enumerate(chunk):
            img  = _tensor_to_numpy(sample["image"])
            lbl  = _tensor_to_numpy(sample["label"])
            pred = _tensor_to_numpy(sample["prediction"])
            row_title = sample.get("title", f"Sample {chunk_start + r + 1}")

            ax_img  = fig.add_subplot(gs[r, 0])
            ax_lbl  = fig.add_subplot(gs[r, 1])
            ax_pred = fig.add_subplot(gs[r, 2])

            ax_img.imshow(img,  cmap="gray" if img.ndim == 2 else None)
            ax_lbl.imshow(lbl,  cmap="gray")
            ax_pred.imshow(pred, cmap="gray")

            ax_img.set_title(f"{row_title} — Input",       fontsize=9)
            ax_lbl.set_title(f"{row_title} — Ground Truth", fontsize=9)
            ax_pred.set_title(f"{row_title} — Prediction",  fontsize=9)

            for ax in (ax_img, ax_lbl, ax_pred):
                ax.axis("off")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _add_uncertainty_page(pdf, uncertainty_dict, suptitle="Uncertainty Maps"):
    """
    uncertainty_dict : dict[str, ndarray]  (keys: mean, std, entropy, …)
    """
    keys = [k for k in ("mean", "std", "entropy", "mutual_information") if k in uncertainty_dict]
    if not keys:
        return

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    cmaps = {
        "mean":                "gray",
        "std":                 "hot",
        "entropy":             "viridis",
        "mutual_information":  "plasma",
        "expected_entropy":    "inferno",
    }

    for ax, key in zip(axes, keys):
        arr = np.asarray(uncertainty_dict[key])
        im = ax.imshow(arr, cmap=cmaps.get(key, "viridis"))
        ax.set_title(key.replace("_", " ").title(), fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_metrics_table_page(pdf, metrics_dict, title="Metrics Summary"):
    """
    metrics_dict : dict[str, float | str]
    """
    if not metrics_dict:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(metrics_dict) + 1.5)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    rows = [[k, f"{v:.6f}" if isinstance(v, float) else str(v)]
            for k, v in metrics_dict.items()]
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1])
    # Header styling
    for col in (0, 1):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_feature_channels_page(pdf, stack_hwc, channel_names=None, max_channels=16,
                                suptitle="Feature Channels"):
    """
    stack_hwc : ndarray (H, W, C) — multichannel geophysical stack.
    Plots up to max_channels channels on a grid.
    """
    stack = np.asarray(stack_hwc)
    if stack.ndim != 3:
        return
    C = stack.shape[2]
    n = min(C, max_channels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    for i in range(n):
        ax = axes[i]
        ch = stack[:, :, i]
        im = ax.imshow(ch, cmap="RdBu_r", aspect="auto")
        name = channel_names[i] if channel_names and i < len(channel_names) else f"ch{i}"
        ax.set_title(name, fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_training_report_pdf(
    output_path,
    *,
    title="Lineament Detection — Training Report",
    subtitle=None,
    report_metadata=None,
    train_losses=None,
    val_losses=None,
    prediction_samples=None,
    uncertainty_dict=None,
    metrics_dict=None,
    stack_hwc=None,
    channel_names=None,
    extra_figures=None,
):
    """
    Generate a self-contained PDF report for a lineament detection experiment.

    Parameters
    ----------
    output_path : str or Path
        Destination PDF file.
    title : str
        Report title shown on the cover page.
    subtitle : str, optional
        Subtitle on the cover page.
    report_metadata : dict, optional
        Key/value pairs shown on the cover page (e.g. model name, date, dataset).
    train_losses : list[float], optional
        Per-epoch training losses.
    val_losses : list[float], optional
        Per-epoch validation losses. Must be same length as train_losses.
    prediction_samples : list[dict], optional
        Each dict must have keys 'image', 'label', 'prediction'
        (torch.Tensor or ndarray) and optionally 'title'.
    uncertainty_dict : dict[str, ndarray], optional
        Uncertainty maps keyed by 'mean', 'std', 'entropy', 'mutual_information'.
    metrics_dict : dict[str, float], optional
        Final evaluation metrics (e.g. IoU, F1, OIS).
    stack_hwc : ndarray (H, W, C), optional
        A representative geophysical stack to visualise its channels.
    channel_names : list[str], optional
        Names for each channel in stack_hwc.
    extra_figures : list[matplotlib.figure.Figure], optional
        Any additional pre-built figures to append at the end.

    Returns
    -------
    str
        Absolute path of the saved PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        _add_title_page(pdf, title, subtitle=subtitle, metadata=report_metadata)

        if train_losses is not None and val_losses is not None:
            _add_loss_curves_page(pdf, train_losses, val_losses)

        if metrics_dict:
            _add_metrics_table_page(pdf, metrics_dict)

        if prediction_samples:
            _add_prediction_grid_page(pdf, prediction_samples)

        if uncertainty_dict:
            _add_uncertainty_page(pdf, uncertainty_dict)

        if stack_hwc is not None:
            _add_feature_channels_page(pdf, stack_hwc, channel_names=channel_names)

        if extra_figures:
            for fig in extra_figures:
                pdf.savefig(fig, bbox_inches="tight")

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = title
        d["Subject"] = "Geophysical lineament detection experiment report"
        d["Creator"] = "lineament-detection library"

    return str(output_path.resolve())
