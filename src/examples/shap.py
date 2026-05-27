"""Feature attribution analysis for the lineament-detection UNet.

Three methods that work without extra dependencies:
  1. Gradient x Input  — fast saliency; differentiable path through the model
  2. Integrated Gradients — more stable than plain gradients; zero baseline
  3. Permutation Importance — model-agnostic; shuffles one channel at a time

Optional (requires ``shap`` package):
  4. SHAP DeepExplainer — Shapley values; needs a separate background batch

Usage
-----
    python src/examples/shap.py \\
        --checkpoint  data/Colombia/checkpoints/colombia_finetuned_unet_v3.pt \\
        --stacks_dir  data/Colombia/training/stacks \\
        --n_channels  5 \\
        --channel_names dem gravity_tga magnetic_tilt dem_slope gravity_residual \\
        --n_tiles     20 \\
        --tile_size   256 \\
        --out_dir     results/attribution

    # With SHAP (slower, needs `pip install shap`):
        --use_shap --n_background 10
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Allow `python src/examples/shap.py` from the project root
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model.Org_UNet import UNet


def _to_numpy(tensor):
    """Convert a CPU tensor to ndarray without using the torch↔numpy bridge.

    PyTorch compiled against NumPy 1.x raises 'Numpy is not available' when
    NumPy 2.x is installed and `.numpy()` is called.  Going through
    `.tolist()` (pure Python) avoids that bridge entirely.
    """
    return np.array(tensor.detach().cpu().tolist(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stack_tiles(stacks_dir, n_channels, n_tiles=20, tile_size=256, seed=42):
    """
    Load random square tiles from .npy stack files (shape H x W x C).

    Returns a list of float32 tensors of shape (1, C, tile_size, tile_size),
    each z-score normalised per channel.
    """
    rng   = np.random.default_rng(seed)
    paths = sorted(Path(stacks_dir).glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy stacks found in {stacks_dir}")

    tiles   = []
    indices = rng.choice(len(paths), size=n_tiles * 4, replace=True)

    for idx in indices:
        if len(tiles) >= n_tiles:
            break
        stack = np.load(paths[idx])          # (H, W, C) expected
        if stack.ndim == 2:
            stack = stack[..., np.newaxis]
        if stack.shape[-1] < n_channels:
            continue
        stack = stack[..., :n_channels].astype(np.float32)

        h, w = stack.shape[:2]
        if h < tile_size or w < tile_size:
            continue

        r = int(rng.integers(0, h - tile_size))
        c = int(rng.integers(0, w - tile_size))
        tile = stack[r:r + tile_size, c:c + tile_size, :].transpose(2, 0, 1)  # (C, H, W)

        for ch in range(tile.shape[0]):
            std = tile[ch].std()
            if std > 0:
                tile[ch] = (tile[ch] - tile[ch].mean()) / std

        tiles.append(torch.tensor(tile, dtype=torch.float32).unsqueeze(0))  # (1, C, H, W)

    if not tiles:
        raise ValueError(
            f"Could not extract any {tile_size}x{tile_size} tiles from {stacks_dir}. "
            "Check --tile_size and --n_channels."
        )
    return tiles


# ---------------------------------------------------------------------------
# Attribution methods
# ---------------------------------------------------------------------------

def gradient_x_input(model, tiles, device="cpu"):
    """
    |grad(output) / grad(input)| x |input|, averaged over predicted-positive
    pixels and all tiles.  Returns ndarray (C,).
    """
    model.eval().to(device)
    scores = []

    for tile in tiles:
        x = tile.to(device).requires_grad_(True)
        out = torch.sigmoid(model(x))

        # Back-propagate from lineament-predicted pixels; fall back to all pixels
        mask  = (out.detach() > 0.5).float()
        score = (out * mask).sum()
        if score.item() == 0:
            score = out.sum()
        score.backward()

        saliency = (x.grad.detach().abs() * x.detach().abs())  # (1, C, H, W)
        scores.append(_to_numpy(saliency.mean(dim=(0, 2, 3))))

    return np.mean(scores, axis=0)


def integrated_gradients(model, tiles, steps=50, device="cpu"):
    """
    Integrated Gradients from a zero baseline.

    IG_c = mean_pixel |  (x_c - 0) * mean_alpha( d sigmoid(f) / d x_c )  |

    Returns ndarray (C,).
    """
    model.eval().to(device)
    alphas = torch.linspace(0, 1, steps, device=device)
    scores = []

    for tile in tiles:
        x        = tile.to(device)
        baseline = torch.zeros_like(x)
        grad_acc = torch.zeros_like(x)

        for alpha in alphas:
            interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            torch.sigmoid(model(interp)).sum().backward()
            grad_acc = grad_acc + interp.grad.detach()

        ig = ((x - baseline) * (grad_acc / steps)).abs()   # (1, C, H, W)
        scores.append(_to_numpy(ig.mean(dim=(0, 2, 3))))

    return np.mean(scores, axis=0)


def permutation_importance(model, tiles, n_repeats=5, device="cpu"):
    """
    For each channel c, spatially shuffle it and measure the mean absolute
    change in sigmoid(output).  Higher change = more important.

    Returns ndarray (C,).
    """
    model.eval().to(device)
    n_channels = tiles[0].shape[1]
    importance = np.zeros(n_channels)

    for tile in tiles:
        x = tile.to(device)
        with torch.no_grad():
            base_out = torch.sigmoid(model(x)).detach()

        for c in range(n_channels):
            delta = 0.0
            for _ in range(n_repeats):
                x_perm = x.clone()
                _, _, h, w = x_perm.shape
                perm = torch.randperm(h * w, device=device)
                x_perm[0, c] = x_perm[0, c].reshape(-1)[perm].reshape(h, w)

                with torch.no_grad():
                    perm_out = torch.sigmoid(model(x_perm)).detach()

                delta += (base_out - perm_out).abs().mean().item()

            importance[c] += delta / n_repeats

    return importance / len(tiles)


def shap_channel_importance(model, tiles, n_background=10, device="cpu"):
    """
    SHAP DeepExplainer — requires ``pip install shap``.

    Uses the first ``n_background`` tiles as the reference distribution and
    the remaining tiles as test samples.  Returns ndarray (C,).
    """
    import shap  # noqa: PLC0415

    if len(tiles) <= n_background:
        raise ValueError(
            f"Need > {n_background} tiles for SHAP "
            f"(loaded {len(tiles)}).  Increase --n_tiles or reduce --n_background."
        )

    model.eval().to(device)

    class _SigmoidWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.sigmoid(self.m(x))

    background  = torch.cat(tiles[:n_background], dim=0).to(device)
    test_batch  = torch.cat(tiles[n_background:], dim=0).to(device)

    explainer   = shap.DeepExplainer(_SigmoidWrapper(model), background)
    shap_vals   = explainer.shap_values(test_batch)   # (T, C, H, W) or list

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    return np.abs(np.array(shap_vals)).mean(axis=(0, 2, 3))   # (C,)


# ---------------------------------------------------------------------------
# Spatial maps (Integrated Gradients averaged over tiles)
# ---------------------------------------------------------------------------

def spatial_ig_maps(model, tiles, steps=30, device="cpu"):
    """
    Return per-channel spatial |IG| maps averaged over tiles.

    Returns ndarray (C, H, W).
    """
    model.eval().to(device)
    n_ch = tiles[0].shape[1]
    h, w = tiles[0].shape[2], tiles[0].shape[3]
    accum  = np.zeros((n_ch, h, w), dtype=np.float64)
    alphas = torch.linspace(0, 1, steps, device=device)

    for tile in tiles:
        x        = tile.to(device)
        baseline = torch.zeros_like(x)
        grad_acc = torch.zeros_like(x)

        for alpha in alphas:
            interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            torch.sigmoid(model(interp)).sum().backward()
            grad_acc = grad_acc + interp.grad.detach()

        ig = ((x - baseline) * (grad_acc / steps)).abs()
        accum += _to_numpy(ig[0]).astype(np.float64)

    return accum / len(tiles)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_channel_importance(importances_dict, channel_names, save_path=None):
    """Horizontal bar chart comparing all attribution methods."""
    methods   = list(importances_dict.keys())
    n_methods = len(methods)
    n_ch      = len(channel_names)
    palette   = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(5 * n_methods, max(6, n_ch * 0.38)),
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        vals       = importances_dict[method]
        norm_vals  = vals / (vals.max() + 1e-12)
        order      = np.argsort(norm_vals)
        bar_colors = [palette[i % len(palette)] for i in range(n_ch)]

        ax.barh(
            [channel_names[i] for i in order],
            norm_vals[order],
            color=[bar_colors[i] for i in order],
            edgecolor="white", linewidth=0.4,
        )
        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalised importance", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(0, 1.08)
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle("Channel importance — lineament detection UNet",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_spatial_maps(spatial_maps, channel_names, n_cols=4, save_path=None):
    """Grid of per-channel spatial attribution maps."""
    n_ch   = spatial_maps.shape[0]
    n_rows = (n_ch + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.2))
    axes = np.array(axes).flatten()

    for i in range(n_ch):
        ax = axes[i]
        im = ax.imshow(spatial_maps[i], cmap="hot", aspect="auto")
        ax.set_title(channel_names[i], fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_ch:]:
        ax.axis("off")

    fig.suptitle("Spatial attribution (Integrated Gradients)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_importance_table(importances_dict, channel_names):
    methods = list(importances_dict.keys())
    col_w   = max(len(m) for m in methods) + 4
    header  = f"{'Channel':<32}" + "".join(f"{m:>{col_w}}" for m in methods)
    sep     = "=" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")

    first_vals = importances_dict[methods[0]]
    order      = np.argsort(first_vals)[::-1]

    for i in order:
        row = f"{channel_names[i]:<32}"
        for m in methods:
            row += f"{importances_dict[m][i]:>{col_w}.6f}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature attribution analysis for lineament UNet."
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to model .pt checkpoint")
    parser.add_argument("--stacks_dir",    required=True,
                        help="Directory containing .npy multichannel stacks (H x W x C)")
    parser.add_argument("--n_channels",    type=int,   default=5)
    parser.add_argument("--channel_names", nargs="*",  default=None,
                        help="Channel names in stack order (default: ch_00, ch_01, ...)")
    parser.add_argument("--tile_size",     type=int,   default=256)
    parser.add_argument("--n_tiles",       type=int,   default=20,
                        help="Total tiles to sample (need at least n_background+2 for SHAP)")
    parser.add_argument("--n_ig_steps",    type=int,   default=50,
                        help="Integration steps for Integrated Gradients")
    parser.add_argument("--n_perm_reps",   type=int,   default=5,
                        help="Shuffle repeats per channel for Permutation Importance")
    parser.add_argument("--use_shap",      action="store_true",
                        help="Also run SHAP DeepExplainer (requires pip install shap)")
    parser.add_argument("--n_background",  type=int,   default=10,
                        help="Background tiles for SHAP DeepExplainer")
    parser.add_argument("--spatial_tiles", type=int,   default=8,
                        help="Tiles used for spatial IG maps (fewer = faster)")
    parser.add_argument("--out_dir",       default="results/attribution")
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_names = args.channel_names or [f"ch_{i:02d}" for i in range(args.n_channels)]
    if len(channel_names) != args.n_channels:
        raise ValueError(
            f"{len(channel_names)} channel names provided but --n_channels={args.n_channels}"
        )

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt.get("model_state_dict", ckpt)
    model = UNet(n_classes=1, input_channels=args.n_channels)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"Channels ({args.n_channels}): {channel_names}")

    # Load tiles
    tiles = load_stack_tiles(
        args.stacks_dir, args.n_channels,
        n_tiles=args.n_tiles, tile_size=args.tile_size,
    )
    print(f"Loaded {len(tiles)} tiles  ({args.tile_size}x{args.tile_size})")

    # ---- Gradient x Input ----
    print("\n[1/3] Gradient x Input ...")
    gi = gradient_x_input(model, tiles, device=args.device)

    # ---- Integrated Gradients ----
    print(f"[2/3] Integrated Gradients ({args.n_ig_steps} steps) ...")
    ig = integrated_gradients(model, tiles, steps=args.n_ig_steps, device=args.device)

    # ---- Permutation Importance ----
    print(f"[3/3] Permutation Importance ({args.n_perm_reps} reps x {len(tiles)} tiles) ...")
    pi = permutation_importance(model, tiles, n_repeats=args.n_perm_reps, device=args.device)

    importances = {
        "Gradient x Input":      gi,
        "Integrated Gradients":  ig,
        "Permutation Importance": pi,
    }

    # ---- Optional SHAP ----
    if args.use_shap:
        print(f"[+] SHAP DeepExplainer (background={args.n_background}) ...")
        try:
            sp = shap_channel_importance(
                model, tiles, n_background=args.n_background, device=args.device,
            )
            importances["SHAP DeepExplainer"] = sp
        except Exception as exc:
            print(f"    Warning: SHAP failed — {exc}")

    # ---- Results ----
    print_importance_table(importances, channel_names)

    plot_channel_importance(
        importances, channel_names,
        save_path=out_dir / "channel_importance.png",
    )

    print(f"\n[Spatial] IG maps on {args.spatial_tiles} tiles ...")
    spatial = spatial_ig_maps(
        model, tiles[:args.spatial_tiles], steps=30, device=args.device,
    )
    plot_spatial_maps(
        spatial, channel_names,
        save_path=out_dir / "spatial_attribution.png",
    )

    # Save raw arrays
    save_kwargs = {
        k.replace(" ", "_").replace("x", "x"): v
        for k, v in importances.items()
    }
    np.savez(out_dir / "importances.npz",
             channel_names=np.array(channel_names), **save_kwargs)
    print(f"\nAll results saved to  {out_dir}/")


if __name__ == "__main__":
    main()
