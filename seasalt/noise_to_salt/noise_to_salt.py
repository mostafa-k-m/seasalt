import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.logging import RichHandler
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

root_folder = Path().resolve()
models_folder = root_folder.joinpath("models").resolve()
data_folder = root_folder.joinpath("data").joinpath("train").resolve()

logging.basicConfig(
    format="%(asctime)s %(levelname)-5s " "[%(filename)s:%(lineno)d] %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), RichHandler()],
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
    def __init__(self) -> None:
        super(NoiseDetector, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding="same"),
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

        # # Decoder
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
            torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=4),
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


def eval_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    noise_type: str,
    noise_parameter: float,
    device: torch.device,
) -> float:
    model.eval()
    mse = []
    with torch.no_grad():
        for images, _ in dataloader:
            noisy_images, masks = noise_adder(images, noise_parameter, noise_type)
            images = images.to(device)
            noisy_images = noisy_images.to(device)
            preds = model(images)
            mse.extend(MSE(masks.cpu().detach(), preds.cpu().detach()))
        return np.array(mse).mean()


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


def train_model(
    model: torch.nn.Module,
    noise_type: str,
    noise_parameter: float,
    optimizer: torch.optim.Adam,
    criterion: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    run_name: str,
    num_epochs=100,
) -> None:
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    model.train()
    train_error = 0
    val_error = 0
    for epoch in range(num_epochs):
        for i, (noisy_images, masks) in track(
            enumerate(train_dataloader),
            description=f"train_epoch_#{epoch}",
            total=len(train_dataloader),
        ):
            noisy_images = noisy_images.to(device)
            masks = masks.to(device)
            pred_masks = model(noisy_images)
            loss = criterion(pred_masks, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_error = eval_model(
                model, train_dataloader, noise_type, noise_parameter, device
            )
            val_error = eval_model(
                model, val_dataloader, noise_type, noise_parameter, device
            )
        log_progress(writer, train_error, val_error, epoch, num_epochs, run_name, model)
