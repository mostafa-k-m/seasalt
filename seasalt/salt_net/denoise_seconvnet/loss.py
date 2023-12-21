from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


class MixL1SSIMLoss(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(
        self,
        gaussian_sigmas: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0],
        K: Tuple[float, float] = (0.01, 0.03),
        alpha: float = 0.985,
        channels: int = 1,
    ) -> None:
        super(MixL1SSIMLoss, self).__init__()
        self.channels = channels
        self.C1 = (K[0]) ** 2
        self.C2 = (K[1]) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.stack(
            [
                self._fspecial_gauss_2d(filter_size, sigma)
                for sigma in sorted(gaussian_sigmas * 3)
            ]
        )
        self.g_masks = g_masks.view(-1, 1, filter_size, filter_size).contiguous()

    def _fspecial_gauss_1d(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords.pow(2)) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size: int, sigma: float) -> torch.Tensor:
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g_masks = self.g_masks.to(x.device, dtype=x.dtype)
        mux = F.conv2d(x, g_masks, groups=self.channels, padding=self.pad)
        muy = F.conv2d(y, g_masks, groups=self.channels, padding=self.pad)

        mux2, muy2, muxy = mux.pow(2), muy.pow(2), mux * muy

        sigmax2 = (
            F.conv2d(x * x, g_masks, groups=self.channels, padding=self.pad) - mux2
        )
        sigmay2 = (
            F.conv2d(y * y, g_masks, groups=self.channels, padding=self.pad) - muy2
        )
        sigmaxy = (
            F.conv2d(x * y, g_masks, groups=self.channels, padding=self.pad) - muxy
        )

        ssim_map = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # noqa: E741
        cs_map = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        loss_ms_ssim = 1 - ssim_map[:, -3:, :, :].prod(dim=1) * cs_map.prod(dim=1)

        loss_l1 = F.l1_loss(x, y, reduction="none")
        gaussian_l1 = F.conv2d(
            loss_l1,
            g_masks.narrow(dim=0, start=-3, length=3),
            groups=self.channels,
            padding=self.pad,
        ).mean(1)

        loss_mix = (1 - self.alpha) * loss_ms_ssim + self.alpha * gaussian_l1
        return 100 * loss_mix.mean()


class GradientVarianceLoss(torch.nn.Module):
    def __init__(self, patch_size: int = 8):
        super(GradientVarianceLoss, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = (
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_y = (
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gray_output = tvF.rgb_to_grayscale(output, 1)
        gray_target = tvF.rgb_to_grayscale(target, 1)

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(
            gray_target, self.kernel_x.to(output.device), stride=1, padding=1
        )
        gy_target = F.conv2d(
            gray_target, self.kernel_y.to(output.device), stride=1, padding=1
        )
        gx_output = F.conv2d(
            gray_output, self.kernel_x.to(output.device), stride=1, padding=1
        )
        gy_output = F.conv2d(
            gray_output, self.kernel_y.to(output.device), stride=1, padding=1
        )

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient
        # maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(
            var_target_y, var_output_y
        )

        return gradvar_loss
