import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
# weighted_bce_loss(pred,mask, reduction='mean')+
def bce_iou_loss_logit(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(pred, target, smooth=1e-6):
    # 计算预测值与真实标签的交集
    pred = torch.sigmoid(pred)
    target = target.detach().clone()
    target[target > 0] = 1
    intersection = torch.sum(pred * target, dim=(2, 3))

    # 计算预测值和真实标签的和
    pred_sum = torch.sum(pred, dim=(2, 3))
    target_sum = torch.sum(target, dim=(2, 3))

    # 计算 Dice 系数
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    # 计算 Dice Loss
    loss = 1.0 - dice_coeff

    return loss.mean()


def loss_compute(pred,mask):
    B, _, H, W = mask.shape

    loss=0.0
    final_loss=0.0
    for i in pred:
        i = F.interpolate(i, (H, W), mode='bilinear', align_corners=False)
        loss = weighted_bce_loss_with_logits(i, mask, reduction='mean')+ iou_loss_with_logits(i, mask,reduction='mean')
        binary_dice_loss = dice_loss(i, mask)
        final_loss = loss + binary_dice_loss+final_loss

    return final_loss

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def bce_loss(pred, mask, reduction='none'):
    # print('loss',pred.shape,mask.shape)
    bce = F.binary_cross_entropy(pred, mask, reduction=reduction)
    nan_mask = torch.isnan(bce)

    return bce
def iou_loss(pred, mask, reduction='none'):
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    if reduction == 'mean':
        iou = iou.mean()

    return iou
def weighted_bce_loss(pred, mask, reduction='none'):
    # print('loss',pred.shape,mask.shape)

    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()

    bce = weight * bce_loss(pred, mask, reduction='none').flatten()

    if reduction == 'mean':
        bce = bce.mean()
    # print(bce)
    return bce
def weighted_bce_loss_with_logits(pred, mask, reduction='none'):
    return weighted_bce_loss(torch.sigmoid(pred), mask, reduction=reduction)
def iou_loss_with_logits(pred, mask, reduction='none'):
    return iou_loss(torch.sigmoid(pred), mask, reduction=reduction)