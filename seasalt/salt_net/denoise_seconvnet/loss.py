from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


class MixL1SSIMLoss(torch.nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(
        self,
        gaussian_sigmas: List[float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        K: Tuple[float, float] = (0.01, 0.03),
        alpha: float = 0.84,
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

        ssim_map = torch.relu((2 * muxy + self.C1) / (mux2 + muy2 + self.C1))
        cs_map = torch.relu((2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2))

        loss_ms_ssim = 1 - ssim_map[:, -3:, :, :].prod(dim=1) * cs_map.prod(dim=1)

        loss_l1 = F.l1_loss(x, y, reduction="none")
        gaussian_l1 = F.conv2d(
            loss_l1,
            g_masks.narrow(dim=0, start=-3, length=3),
            groups=self.channels,
            padding=self.pad,
        ).mean(1)

        loss_mix = (
            self.alpha * torch.clamp(torch.nan_to_num(loss_ms_ssim), 0, 1)
            + (1 - self.alpha) * gaussian_l1
        )
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


class PSNRLoss(torch.nn.Module):

    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / torch.log(torch.tensor(10.0)).item()
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )
