import matplotlib.pyplot as plt
import torch

def plot_input_gt(image, label):
    image = image.permute(1, 2, 0) if image.dtype == torch.float else image
    label = label.permute(1, 2, 0) if label.dtype == torch.float else label

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

    ax0.imshow(image, cmap='gray') if image.shape[2] == 1 else ax0.imshow(image)
    ax0.set_title('Input Image')
    ax1.imshow(label, cmap='gray')
    ax1.set_title('Ground Truth')

    fig.tight_layout()

def plot_input_gt_pred(image, label, prediction):
    image = image.permute(1, 2, 0) if image.dtype == torch.float else image
    label = label.permute(1, 2, 0) if label.dtype == torch.float else label
    prediction = prediction.permute(1, 2, 0) if prediction.dtype == torch.float else prediction

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

    ax0.imshow(image, cmap='gray') if image.shape[2] == 1 else ax0.imshow(image)
    ax0.set_title('Input Image')
    ax1.imshow(label, cmap='gray')
    ax1.set_title('Ground Truth')
    ax2.imshow(prediction, cmap='gray')
    ax2.set_title('Prediction')

    fig.tight_layout()