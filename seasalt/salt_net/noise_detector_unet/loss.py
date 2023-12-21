from typing import Optional

import torch
from torch.nn import BCELoss


def dice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    inter = 2 * (input * target).sum(dim=(-1, -2, -3))
    sets_sum = input.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return 1 - dice.mean()


class DiceLoss(BCELoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return BCELoss.forward(self, input, target) + dice_loss(input, target)
