import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None
    Subset = None

try:
    from src.preprocessing.multisource_dataset import MultiSourceStackDataset
    from src.training.uncertainty import mc_dropout_predict
except ModuleNotFoundError:
    from preprocessing.multisource_dataset import MultiSourceStackDataset
    from training.uncertainty import mc_dropout_predict


def create_multisource_dataloaders(
    stack_dir,
    mask_dir,
    batch_size=4,
    val_fraction=0.2,
    seed=42,
    num_workers=0,
    normalize=True,
    drop_last_train=False,
):
    """
    Build reproducible train/val dataloaders from multichannel .npy stacks.

    Parameters
    ----------
    stack_dir, mask_dir : str
        Folders with aligned stack/mask .npy files.
    val_fraction : float
        Fraction in (0, 1) used for validation split.
    """
    if torch is None:
        raise ImportError("torch is required to create dataloaders")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    dataset = MultiSourceStackDataset(
        stack_dir=stack_dir,
        mask_dir=mask_dir,
        normalize=normalize,
    )

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("Not enough samples after split. Add more stack/mask pairs.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}


def predict_stack_with_uncertainty(
    model,
    stack_hwc,
    n_samples=30,
    activation="sigmoid",
    device="cpu",
    pad_multiple=16,
):
    """
    Predict segmentation and uncertainty maps from a single multichannel stack.

    Parameters
    ----------
    model : torch.nn.Module
    stack_hwc : np.ndarray
        Shape (H, W, C)
    pad_multiple : int
        If > 1, pad spatial dimensions to the nearest multiple before inference
        and crop outputs back to original size. Useful for U-Net-like models.

    Returns
    -------
    dict[str, np.ndarray]
        mean, std, entropy, expected_entropy, mutual_information
    """
    if torch is None:
        raise ImportError("torch is required for uncertainty prediction")

    stack = np.asarray(stack_hwc, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape (H, W, C), got {stack.shape}")

    h, w, _ = stack.shape
    if pad_multiple is None:
        pad_multiple = 1
    if pad_multiple < 1:
        raise ValueError("pad_multiple must be >= 1")

    pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
    pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple

    if pad_h or pad_w:
        stack = np.pad(
            stack,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )

    x = torch.tensor(stack.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
    model = model.to(device)

    out = mc_dropout_predict(
        model=model,
        inputs=x,
        n_samples=n_samples,
        activation=activation,
    )

    cropped = {}
    for key, value in out.items():
        value = value[..., :h, :w]
        cropped[key] = value.detach().cpu().numpy().squeeze()

    return cropped
