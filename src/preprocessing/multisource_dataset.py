import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSourceStackDataset(Dataset):
    """
    Dataset for multichannel geophysical stacks and binary masks.

    Expected file format:
    - stack files: .npy arrays with shape (H, W, C)
    - mask files: .npy arrays with shape (H, W) or (H, W, 1)
    """

    def __init__(
        self,
        stack_dir,
        mask_dir,
        stack_suffix=".npy",
        mask_suffix=".npy",
        normalize=True,
        eps=1e-6,
    ):
        self.stack_dir = stack_dir
        self.mask_dir = mask_dir
        self.normalize = normalize
        self.eps = eps

        self.stack_files = sorted([f for f in os.listdir(stack_dir) if f.endswith(stack_suffix)])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(mask_suffix)])

        if len(self.stack_files) != len(self.mask_files):
            raise ValueError(
                f"Mismatch between stacks ({len(self.stack_files)}) and masks ({len(self.mask_files)})"
            )

        stack_roots = [os.path.splitext(f)[0] for f in self.stack_files]
        mask_roots = [os.path.splitext(f)[0] for f in self.mask_files]
        if stack_roots != mask_roots:
            raise ValueError("Stack and mask filenames must match exactly (without extension)")

    def __len__(self):
        return len(self.stack_files)

    def _normalize_stack(self, stack):
        # Channel-wise z-score normalization
        mean = stack.mean(axis=(0, 1), keepdims=True)
        std = stack.std(axis=(0, 1), keepdims=True)
        return (stack - mean) / (std + self.eps)

    def __getitem__(self, idx):
        stack_path = os.path.join(self.stack_dir, self.stack_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        stack = np.load(stack_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if stack.ndim != 3:
            raise ValueError(f"Expected stack shape (H, W, C), got {stack.shape} in {stack_path}")

        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H, W) or (H, W, 1), got {mask.shape} in {mask_path}")

        if stack.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Spatial mismatch between stack {stack.shape[:2]} and mask {mask.shape[:2]}"
            )

        if self.normalize:
            stack = self._normalize_stack(stack)

        stack_tensor = torch.tensor(stack.transpose(2, 0, 1), dtype=torch.float32)
        mask_tensor = torch.tensor(mask[None, ...], dtype=torch.float32)

        return stack_tensor, mask_tensor
