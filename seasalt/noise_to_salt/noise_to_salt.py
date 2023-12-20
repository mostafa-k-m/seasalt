import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
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


def calculate_kernel(
    size: int,
    exp: int = 1,
) -> NDArray[np.float64]:
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
    return distance_weights.reshape(size, size)


def weighted_mean_conv(img: NDArray[np.float64], size: int = 3) -> torch.Tensor:
    kernel = calculate_kernel(size)
    img_tensor = torch.tensor(img, dtype=torch.float32)
    weighted_average_kernel = torch.tensor(kernel, dtype=torch.float32)
    weighted_average_kernel = weighted_average_kernel / torch.sum(
        weighted_average_kernel
    )
    img_tensor = img_tensor.unsqueeze(0)
    weighted_average_kernel = weighted_average_kernel.unsqueeze(0).unsqueeze(0)
    result = F.conv2d(
        img_tensor, weighted_average_kernel, stride=1, padding=int((size - 1) / 2)
    )
    result = result.squeeze()
    return torch.abs(result - img)


def weighted_mean_conv_rgb(img: NDArray[np.float64], size: int = 3) -> torch.Tensor:
    kernel = calculate_kernel(size)
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1)  # Change channel order
    result_channels = []

    for channel in range(3):
        current_channel = img_tensor[channel, :, :]
        weighted_average_kernel = torch.tensor(kernel, dtype=torch.float32)
        weighted_average_kernel = weighted_average_kernel / torch.sum(
            weighted_average_kernel
        )
        current_channel = current_channel.unsqueeze(0).unsqueeze(0)
        weighted_average_kernel = weighted_average_kernel.unsqueeze(0).unsqueeze(0)
        result = F.conv2d(
            current_channel,
            weighted_average_kernel,
            stride=1,
            padding=int((size - 1) / 2),
        )
        result_channels.append(result.squeeze())

    result = torch.stack(result_channels)
    result = result.permute(1, 2, 0)  # Change channel order back
    return torch.abs(result - img_tensor.permute(1, 2, 0))


class NoiseDetector(torch.nn.Module):
    def __init__(self, channels=1) -> None:
        super(NoiseDetector, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d((2, 2), padding=0),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d((2, 2), padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d((2, 2), padding=0),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, channels, kernel_size=3, stride=1, padding=4),
            torch.nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = self.decoder(x)
        return x


def PSNR(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(torch.mean((preds - target) ** 2, dim=[1, 2, 3]) + 1e-8)


def MSE(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((preds - target) ** 2, dim=[1, 2, 3])


def noise_adder(
    images: torch.Tensor, noise_parameter: float, noise_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_type == "gaussian":
        noisy_images, masks = _gaussian_noise_adder(images, noise_parameter)
    elif noise_type == "salt and pepper":
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


def log_progress(
    writer: SummaryWriter,
    train_error: float,
    val_error: float,
    epoch: int,
    num_epochs: int,
    run_name: str,
    model: torch.nn.Module,
) -> None:
    writer.add_scalar("validation loss", train_error, epoch)
    writer.add_scalar("train loss", val_error, epoch)
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


def train_model(
    model: torch.nn.Module,
    learning_rate: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    run_name: str,
    num_epochs=100,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    criterion = BCELoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for _, (noisy_images, masks) in track(
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            optimizer.step()

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for noisy_images, masks in val_dataloader:
                noisy_images = noisy_images.to(device)
                masks = masks.to(device)
                pred_masks = model(noisy_images)
                val_loss = criterion(pred_masks, masks)
                val_loss += dice_loss(pred_masks, masks)
                epoch_val_losses.append(val_loss)
        epoch_valid_loss_value = torch.mean(torch.stack(epoch_val_losses)).item()
        scheduler.step(epoch_valid_loss_value)
        log_progress(writer, train_loss, val_loss, epoch, num_epochs, run_name, model)
