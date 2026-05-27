#!/usr/bin/env python
"""
Train a model on Colombian data with train/val split and test on held-out tiles.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model.Org_UNet import UNet
from training.train import train_model
from training.multisource_pipeline import create_multisource_dataloaders


def infer_tile(model, stack_path, device='cpu'):
    """Load a tile, normalize and run inference. Returns prediction probability."""
    stack = np.load(stack_path).astype(np.float32)  # (H, W, 5)
    # Normalize per channel
    for i in range(stack.shape[2]):
        band = stack[:, :, i]
        valid = band[np.isfinite(band)]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            std_val = np.std(valid)
            if std_val > 0:
                stack[:, :, i] = (band - mean_val) / std_val
    # NaN → 0
    stack = np.nan_to_num(stack, 0)
    # (H, W, 5) → (1, 5, H, W)
    x = torch.from_numpy(np.transpose(stack, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, 1, H, W)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
    return probs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and test on Colombian data")
    parser.add_argument("--stack-dir", default="data/Colombia/training/stacks")
    parser.add_argument("--mask-dir", default="data/Colombia/training/masks")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint-out", default="data/Colombia/checkpoints/colombia_trained_local.pt")
    parser.add_argument("--history-out", default="data/Colombia/checkpoints/colombia_trained_local_history.json")
    parser.add_argument("--n-test-tiles", type=int, default=5, help="Number of tiles to hold out for testing")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stack_dir = Path(args.stack_dir)
    mask_dir = Path(args.mask_dir)

    # Load all files
    stack_files = sorted(list(stack_dir.glob("*.npy")))
    mask_files = sorted(list(mask_dir.glob("*.npy")))

    print(f"Found {len(stack_files)} stacks and {len(mask_files)} masks")

    if len(stack_files) != len(mask_files):
        print(f"WARNING: Mismatch in tile counts!")

    # Split into train/val/test
    n_tiles = len(stack_files)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_tiles)
    rng.shuffle(indices)

    n_test = args.n_test_tiles
    n_val = max(1, n_tiles // 5)
    n_train = n_tiles - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Extract test tiles
    test_stacks = [stack_files[i] for i in test_idx]
    test_masks = [mask_files[i] for i in test_idx]

    # Create temporary directory with train+val tiles
    working_stack_tmp = Path("tmp_colombia_working_stacks")
    working_mask_tmp = Path("tmp_colombia_working_masks")
    working_stack_tmp.mkdir(exist_ok=True)
    working_mask_tmp.mkdir(exist_ok=True)

    import shutil

    for i in list(train_idx) + list(val_idx):
        shutil.copy(stack_files[i], working_stack_tmp / stack_files[i].name)
        shutil.copy(mask_files[i], working_mask_tmp / mask_files[i].name)

    # Create dataloaders using built-in val_fraction split
    dataloaders = create_multisource_dataloaders(
        stack_dir=str(working_stack_tmp),
        mask_dir=str(working_mask_tmp),
        batch_size=args.batch_size,
        val_fraction=0.2,
        seed=args.seed,
        num_workers=0,
        normalize=True,
    )

    sample_x, _ = next(iter(dataloaders["train"]))
    in_channels = sample_x.shape[1]

    model = UNet(n_classes=1, input_channels=in_channels)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    trained_model, train_losses, val_losses, best_loss = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience,
        min_delta=1e-4,
    )

    checkpoint_path = Path(args.checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "input_channels": in_channels,
            "best_loss": float(best_loss),
            "epochs": args.epochs,
        },
        checkpoint_path,
    )

    history = {
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "best_loss": float(best_loss),
        "checkpoint": str(checkpoint_path),
        "input_channels": in_channels,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }

    if args.history_out:
        history_path = Path(args.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TESTING ON HELD-OUT TILES")
    print("=" * 70)

    trained_model.eval()

    test_metrics = {}
    for i, (stack_path, mask_path) in enumerate(zip(test_stacks, test_masks)):
        print(f"\nTest tile {i+1}/{len(test_stacks)}: {stack_path.stem}")

        pred_prob = infer_tile(trained_model, stack_path, device=device)
        gt_mask = np.load(mask_path).astype(np.float32)

        # Compute metrics at threshold 0.5
        pred_binary = (pred_prob >= 0.5).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)

        tp = np.sum(gt_binary * pred_binary)
        fp = np.sum((1 - gt_binary) * pred_binary)
        fn = np.sum(gt_binary * (1 - pred_binary))
        tn = np.sum((1 - gt_binary) * (1 - pred_binary))

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        test_metrics[stack_path.stem] = {
            "threshold": 0.5,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "pred_positive_fraction": float(np.mean(pred_binary)),
            "gt_positive_fraction": float(np.mean(gt_binary)),
        }

        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")

    # Summary stats
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    avg_precision = np.mean([m["precision"] for m in test_metrics.values()])
    avg_recall = np.mean([m["recall"] for m in test_metrics.values()])
    avg_f1 = np.mean([m["f1"] for m in test_metrics.values()])
    avg_iou = np.mean([m["iou"] for m in test_metrics.values()])

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

    test_results = {
        "checkpoint": str(checkpoint_path),
        "test_tiles": test_metrics,
        "summary": {
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "avg_iou": float(avg_iou),
            "n_test_tiles": len(test_idx),
        },
    }

    test_results_path = checkpoint_path.parent / f"{checkpoint_path.stem}_test_results.json"
    with test_results_path.open("w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Test results saved: {test_results_path}")
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"✓ History saved: {history_path if args.history_out else 'N/A'}")

    # Cleanup tmp dir
    import shutil

    shutil.rmtree(working_stack_tmp, ignore_errors=True)
    shutil.rmtree(working_mask_tmp, ignore_errors=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
