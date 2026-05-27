import torch
import torch.nn as nn


def _enable_dropout_in_eval(model):
    """Set dropout layers to train mode while keeping the rest in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_predict(
    model,
    inputs,
    n_samples=30,
    activation="sigmoid",
):
    """
    Monte-Carlo Dropout inference for segmentation outputs.

    Parameters
    ----------
    model : torch.nn.Module
    inputs : torch.Tensor
        Shape (B, C, H, W)
    n_samples : int
        Number of stochastic forward passes.
    activation : str
        'sigmoid', 'softmax' or 'none'.

    Returns
    -------
    dict[str, torch.Tensor]
        mean, std, entropy, expected_entropy, mutual_information
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    was_training = model.training
    _enable_dropout_in_eval(model)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(inputs)
            if activation == "sigmoid":
                probs = torch.sigmoid(logits)
            elif activation == "softmax":
                probs = torch.softmax(logits, dim=1)
            elif activation == "none":
                probs = logits
            else:
                raise ValueError("activation must be one of: sigmoid, softmax, none")
            preds.append(probs)

    pred_stack = torch.stack(preds, dim=0)  # (S, B, C, H, W)
    mean_pred = pred_stack.mean(dim=0)
    std_pred = pred_stack.std(dim=0)

    eps = 1e-8
    if activation == "sigmoid":
        entropy = -(
            mean_pred * torch.log(mean_pred + eps)
            + (1 - mean_pred) * torch.log(1 - mean_pred + eps)
        )
        expected_entropy = -(
            pred_stack * torch.log(pred_stack + eps)
            + (1 - pred_stack) * torch.log(1 - pred_stack + eps)
        ).mean(dim=0)
    elif activation == "softmax":
        entropy = -(mean_pred * torch.log(mean_pred + eps)).sum(dim=1, keepdim=True)
        expected_entropy = -(pred_stack * torch.log(pred_stack + eps)).sum(dim=2, keepdim=True).mean(dim=0)
    else:
        entropy = torch.zeros_like(mean_pred)
        expected_entropy = torch.zeros_like(mean_pred)

    mutual_information = entropy - expected_entropy

    if was_training:
        model.train()
    else:
        model.eval()

    return {
        "mean": mean_pred,
        "std": std_pred,
        "entropy": entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }
