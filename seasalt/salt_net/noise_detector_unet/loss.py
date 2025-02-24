from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import BCELoss


class DiceLoss(BCELoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def dice_loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        input = torch.sigmoid(input)
        input = input.view(target.shape[0], -1)
        target = target.view(target.shape[0], -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + epsilon) / (
            input.sum(1) + target.sum(1) + epsilon
        )
        return 1 - dice.sum() / target.shape[0]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.binary_cross_entropy(input, target) + self.dice_loss(
            input, target
        )
