import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.logging import RichHandler
from skimage.metrics import structural_similarity

root_folder = Path().resolve()
models_folder = root_folder.joinpath("models").resolve()

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


def PSNR(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        -10 * torch.log10(torch.mean((preds - target) ** 2, dim=[1, 2, 3]) + 1e-8)
    )


def MSE(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((preds - target) ** 2, dim=[1, 2, 3])


def rgb_to_gray(tensor: torch.Tensor) -> torch.Tensor:
    return (
        0.2989 * tensor[:, 0:1, :, :]
        + 0.5870 * tensor[:, 1:2, :, :]
        + 0.1140 * tensor[:, 2:, :, :]
    )


def SSIM(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    preds = preds * 255
    target = target * 255
    if preds.shape[1] == 1:
        pred_batch = preds.to(dtype=torch.uint8).squeeze().cpu().numpy()
        target_batch = target.to(dtype=torch.uint8).squeeze().cpu().numpy()
    elif preds.shape[1] == 3:
        pred_batch = rgb_to_gray(preds).to(dtype=torch.uint8).squeeze().cpu().numpy()
        target_batch = rgb_to_gray(target).to(dtype=torch.uint8).squeeze().cpu().numpy()
    else:
        raise
    ssim_values = []
    for pred_image, target_image in zip(pred_batch, target_batch):
        ssim_value = structural_similarity(pred_image, target_image)
        ssim_values.append(ssim_value)
    return torch.tensor(ssim_values).mean()


def log_images_to_tensorboard(writer, epoch, input_images, target_images, pred_images):
    writer.add_images("Input Images", input_images, epoch)
    writer.add_images("Predicted", pred_images, epoch)
    writer.add_images("Target", target_images, epoch)


def log_test_to_tensor_board(writer, epoch, val_loss, ssim, psnr):
    writer.add_scalar("valid loss", val_loss, epoch)
    writer.add_scalar(
        "SSIM",
        ssim,  # type:ignore
        epoch,
    )
    writer.add_scalar(
        "PSNR",
        psnr,  # type:ignore
        epoch,
    )


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


def save_model_weights(model, run_name, epoch):
    torch.save(
        model.state_dict(),
        models_folder.joinpath(f"pytorch_{run_name}_{epoch:d}.h5"),
    )
