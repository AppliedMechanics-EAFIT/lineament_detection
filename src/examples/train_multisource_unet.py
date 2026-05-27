import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.model.Org_UNet import UNet
    from src.training.train import train_model
    from src.training.multisource_pipeline import create_multisource_dataloaders
    from src.preprocessing.multisource_dataset import MultiSourceStackDataset
except ModuleNotFoundError:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from model.Org_UNet import UNet
    from training.train import train_model
    from training.multisource_pipeline import create_multisource_dataloaders
    from preprocessing.multisource_dataset import MultiSourceStackDataset


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on prepared multisource .npy tiles")
    parser.add_argument("--stack-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="Trade-off in [0,1]: BCE/focal contribution vs dice contribution.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=["focal", "bce"],
        default="focal",
        help="Use focal+Dice (default) or BCEWithLogits+Dice.",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive class weight for BCEWithLogits when --loss-mode=bce.",
    )
    parser.add_argument(
        "--lambda-geo",
        type=float,
        default=0.0,
        help="Regularization weight for geometric smoothness terms.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--history-out", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint (.pt/.pth) to initialize model weights for fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Epochs without val-loss improvement before early stop. Omit to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum val-loss decrease to count as improvement (default: 1e-4).",
    )
    parser.add_argument("--extra-stack-dirs", nargs='*', default=[])
    parser.add_argument("--extra-mask-dirs", nargs='*', default=[])

    args = parser.parse_args()

    if len(args.extra_stack_dirs) != len(args.extra_mask_dirs):
        parser.error("--extra-stack-dirs and --extra-mask-dirs must have the same number of arguments")

    stack_dirs = [args.stack_dir] + args.extra_stack_dirs
    mask_dirs = [args.mask_dir] + args.extra_mask_dirs

    datasets = []
    for sd, md in zip(stack_dirs, mask_dirs):
        datasets.append(MultiSourceStackDataset(
            stack_dir=sd,
            mask_dir=md,
            normalize=True,
        ))
    dataset = ConcatDataset(datasets)

    n_total = len(dataset)
    n_val = int(args.val_fraction * n_total)
    n_train = n_total - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    dataloaders = {"train": train_loader, "val": val_loader}

    sample_x, _ = next(iter(dataloaders["train"]))
    in_channels = sample_x.shape[1]

    model = UNet(n_classes=1, input_channels=in_channels)

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")

        checkpoint = torch.load(init_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print(f"Initialized model from checkpoint: {init_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)

    trained_model, train_losses, val_losses, best_loss = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
        bce_weight=args.bce_weight,
        use_focal=(args.loss_mode == "focal"),
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight,
        lambda_geo=args.lambda_geo,
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

    history = _to_builtin({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "input_channels": in_channels,
        "loss_config": {
            "mode": args.loss_mode,
            "bce_weight": args.bce_weight,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "pos_weight": args.pos_weight,
            "lambda_geo": args.lambda_geo,
        },
    })
    if args.history_out:
        history_path = Path(args.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", encoding="utf-8") as file_obj:
            json.dump(history, file_obj, indent=2)

    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
