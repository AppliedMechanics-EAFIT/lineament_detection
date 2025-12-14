import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = x.numpy()
            if x.ndim == 3:
                x = np.transpose(x, (1, 2, 0))
            if x.shape[2] == 1:
                x = x[:, :, 0]
        return x

def plot_input_gt(image, label):

    image_np = tensor_to_numpy(image)
    label_np = tensor_to_numpy(label)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

    ax0.imshow(image_np, cmap='gray') if image_np.ndim == 2 else ax0.imshow(image_np)
    ax0.set_title('Input Image')

    ax1.imshow(label_np, cmap='gray') if label_np.ndim == 2 else ax1.imshow(label_np)
    ax1.set_title('Ground Truth')

    fig.tight_layout()

def plot_input_gt_pred(image, label, prediction, vmin=None, vmax=None):
    image = image.permute(1, 2, 0) if image.dtype == torch.float else image
    label = label.permute(1, 2, 0) if label.dtype == torch.float else label
    prediction = prediction.permute(1, 2, 0) if prediction.dtype == torch.float else prediction

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

    if vmin is not None and vmax is not None:
        ax0.imshow(image, cmap='gray', vmin=vmin, vmax=vmax) if image.shape[2] == 1 else ax0.imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax0.imshow(image, cmap='gray') if image.shape[2] == 1 else ax0.imshow(image)

    ax0.set_title('Input Image')
    ax1.imshow(label, cmap='gray')
    ax1.set_title('Ground Truth')
    ax2.imshow(prediction, cmap='gray')
    ax2.set_title('Prediction')

    fig.tight_layout()