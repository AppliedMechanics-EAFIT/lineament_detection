import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def focal_loss_with_logits(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    
    probs = F.sigmoid(inputs)
    
    pt = torch.where(targets == 1, probs, 1 - probs)
    
    focal_factor = (1 - pt) ** gamma
    
    loss = alpha * focal_factor * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
def total_variation_loss(pred):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    return (dx.mean() + dy.mean())

def curvature_loss(pred):
    # First-order differences
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    # Second-order differences (curvature)
    ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    return (ddx.abs().mean() + ddy.abs().mean())

def calc_loss(pred, target, metrics, bce_weight=0.5, use_focal=True, lambda_geo=0.0):
    if use_focal:
        bce = focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0)
    else:
        bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    geom = 0.7 * curvature_loss(pred) + 0.3 * total_variation_loss(pred)
    
    loss = bce * bce_weight + dice * (1 - bce_weight) + lambda_geo * geom

    metrics['bce'] += bce.detach().cpu().numpy() * target.size(0)
    metrics['dice'] += dice.detach().cpu().numpy() * target.size(0)
    metrics['geo']  += geom.detach().cpu().numpy() * target.size(0)
    metrics['loss'] += loss.detach().cpu().numpy() * target.size(0)

    return loss
