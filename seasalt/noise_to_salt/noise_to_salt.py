import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from numpy.typing import NDArray
from rich.logging import RichHandler
from rich.progress import track
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

root_folder = Path().resolve()
models_folder = root_folder.joinpath("models").resolve()
data_folder = root_folder.joinpath("data").joinpath("train").resolve()

logging.basicConfig(
    format="%(asctime)s %(levelname)-5s " "[%(filename)s:%(lineno)d] %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[RichHandler()],
)

logger = logging.getLogger()


def calculate_wa_kernel(
    size: int,
    exp: int = 1,
) -> torch.Tensor:
    all_kernel_positions = np.transpose(np.where(np.ones((size, size)) > 0))
    center_ix = int((size - 1) / 2)
    distance_weights = (
        1
        / (
            1
            + (all_kernel_positions[:, 0] - center_ix) ** 2
            + (all_kernel_positions[:, 1] - center_ix) ** 2
        )
        ** exp
    )
    kernel = distance_weights.reshape(size, size)
    weighted_average_kernel = torch.tensor(kernel, dtype=torch.float32)
    weighted_average_kernel = weighted_average_kernel / torch.sum(
        weighted_average_kernel
    )
    return weighted_average_kernel.unsqueeze(0).unsqueeze(0)


def weighted_mean_conv(img_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    result = F.conv2d(
        img_tensor, kernel, stride=1, padding=int((kernel.shape[-1] - 1) / 2)
    )
    return torch.abs(result - img_tensor)


def weighted_mean_conv_rgb(
    img: NDArray[np.float64], kernel: torch.Tensor
) -> torch.Tensor:
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1)  # Change channel order
    result_channels = []

    for channel in range(3):
        current_channel = img_tensor[channel, :, :]
        current_channel = current_channel.unsqueeze(0).unsqueeze(0)
        result = F.conv2d(
            current_channel,
            kernel,
            stride=1,
            padding=int((kernel.shape[-1] - 1) / 2),
        )
        result_channels.append(result.squeeze())

    result = torch.stack(result_channels)
    result = result.permute(1, 2, 0)  # Change channel order back
    return torch.abs(result - img_tensor.permute(1, 2, 0))


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(EncoderBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )
        self.pooling = torch.nn.MaxPool2d((2, 2), padding=0)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(images)
        pooling_out = self.pooling(conv_out)
        return pooling_out, conv_out


class MiddleBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(MiddleBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, None]:
        conv_out = self.conv(images)
        return conv_out, None


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(DecoderBlock, self).__init__()

        self.t_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2, padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(2 * out_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

    def forward(
        self, x: torch.Tensor, skipped_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.t_conv(x)
        conv_out = self.conv(torch.cat((x, skipped_x), 1))
        return conv_out


class NoiseDetector(torch.nn.Module):
    def __init__(self, channels=1, first_output=64, depth=5) -> None:
        super(NoiseDetector, self).__init__()
        self.encoder = torch.nn.ModuleList(
            [EncoderBlock(channels, first_output)]
            + [
                EncoderBlock(first_output * (2 ** (d - 1)), first_output * (2**d))
                for d in range(1, depth - 1)
            ]
            + [
                MiddleBlock(
                    first_output * (2 ** (depth - 2)), first_output * (2 ** (depth - 1))
                )
            ]
        )

        self.decoder = torch.nn.ModuleList(
            [
                DecoderBlock(first_output * (2**d), first_output * (2 ** (d - 1)))
                for d in range(1, depth)
            ][::-1]
        )

        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                first_output, channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder_outs = []
        x = images

        for module in self.encoder:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.decoder):
            before_pool = encoder_outs[-(i + 2)]
            x = module(x, before_pool)

        return self.output(x)


def PSNR(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(torch.mean((preds - target) ** 2, dim=[1, 2, 3]) + 1e-8)


def MSE(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((preds - target) ** 2, dim=[1, 2, 3])


def noise_adder(
    images: torch.Tensor, noise_parameter: float, noise_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_type == "gauss":
        noisy_images, masks = _gaussian_noise_adder(images, noise_parameter)
    elif noise_type == "sap":
        noisy_images, masks = _salt_and_pepper_noise_adder(images, noise_parameter)
    elif noise_type == "bernoulli":
        noisy_images, masks = _bernoulli_noise_adder(images, noise_parameter)
    elif noise_type == "poisson":
        noisy_images, masks = _poisson_noise_adder(images, noise_parameter)
    else:
        noisy_images, masks = _gaussian_noise_adder(images, noise_parameter)
    return noisy_images, masks.float()


def _gaussian_noise_adder(
    images: torch.Tensor, noise_parameter: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.normal(0, noise_parameter, images.shape)
    noisy_images = (images + noise).clip(0, 1)
    return noisy_images, noise > 0


def _salt_and_pepper_noise_adder(
    images: torch.Tensor, noise_parameter: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    salt_mask = torch.rand(images.shape) < (noise_parameter / 2)
    images = images.masked_fill(salt_mask, 1.0)
    pepper_mask = torch.rand(images.shape) < (noise_parameter / 2)
    images = images.masked_fill(pepper_mask, 0.0)
    return images, torch.logical_or(salt_mask, pepper_mask)


def _bernoulli_noise_adder(
    images: torch.Tensor, noise_parameter: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = noise_parameter * torch.ones(images.shape)
    noise = torch.bernoulli(a)
    noisy_images = images * noise
    return noisy_images, noise > 0


def _poisson_noise_adder(
    images: torch.Tensor, noise_parameter: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = noise_parameter * torch.ones(images.shape)
    p = torch.poisson(a)
    noise = p / p.max()
    noisy_images = (images + noise).clip(0, 1)
    return noisy_images, noise > 0


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


def log_progress_to_console(
    train_error: float,
    val_error: float,
    epoch: int,
    num_epochs: int,
    run_name: str,
    model: torch.nn.Module,
) -> None:
    if epoch % 5 == 0:
        logger.info(
            "Epoch %s/%s:\n train loss: %.5f, validation: %.5f",
            epoch,
            num_epochs,
            train_error,
            val_error,
        )
    if epoch % 100 == 0:
        torch.save(
            model.state_dict(),
            models_folder.joinpath(f"pytorch_{run_name}_{int(epoch/100):d}.h5"),
        )


def log_images_to_tensorboard(model, writer, epoch, noisy_images, masks, pred_masks):
    if epoch % 5 == 0:
        noisy_images_grid = torchvision.utils.make_grid(noisy_images)
        writer.add_image("Input Images", noisy_images_grid, epoch)
        writer.add_graph(model, noisy_images)
        pred_grid = torchvision.utils.make_grid(pred_masks)
        writer.add_image("Predicted", pred_grid, epoch)
        writer.add_graph(model, pred_masks)
        target_grid = torchvision.utils.make_grid(masks)
        writer.add_image("Target", target_grid, epoch)
        writer.add_graph(model, masks)


def train_model(
    model: torch.nn.Module,
    learning_rate: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    run_name: str,
    num_epochs: int = 100,
    log_images: bool = False,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = BCELoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        epoch_train_losses = []
        for step, (noisy_images, masks) in track(
            enumerate(train_dataloader),
            description=f"train_epoch_#{epoch}",
            total=len(train_dataloader),
        ):
            noisy_images = noisy_images.to(device)
            masks = masks.to(device)
            pred_masks = model(noisy_images)
            train_loss = criterion(pred_masks, masks)
            train_loss += dice_loss(pred_masks, masks)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_losses.append(train_loss)
            writer.add_scalar(
                "train loss", train_loss, epoch, len(train_dataloader) * epoch + step
            )

        epoch_val_losses = []
        with torch.no_grad():
            model.eval()
            for step, (noisy_images, masks) in enumerate(val_dataloader):
                noisy_images = noisy_images.to(device)
                masks = masks.to(device)
                pred_masks = model(noisy_images)
                val_loss = criterion(pred_masks, masks)
                val_loss += dice_loss(pred_masks, masks)
                epoch_val_losses.append(val_loss)
                writer.add_scalar(
                    "valid loss", val_loss, epoch, len(val_dataloader) * epoch + step
                )
            log_images_to_tensorboard(
                model, writer, epoch, noisy_images, masks, pred_masks  # type: ignore
            )
        epoch_train_loss_value = torch.mean(torch.stack(epoch_train_losses)).item()
        epoch_valid_loss_value = torch.mean(torch.stack(epoch_val_losses)).item()
        scheduler.step(epoch_valid_loss_value)
        log_progress_to_console(
            epoch_train_loss_value,
            epoch_valid_loss_value,
            epoch,
            num_epochs,
            run_name,
            model,
        )
