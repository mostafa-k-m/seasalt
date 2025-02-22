from typing import Optional

import torch
import torch.optim as optim
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .denoise_seconvnet.loss import MixL1SSIMLoss, PSNRLoss
from .denoise_seconvnet.model import SeConvDesnoiseNet
from .utils import (
    PSNR,
    SSIM,
    log_images_to_tensorboard,
    log_progress_to_console,
    log_test_to_tensor_board,
    save_model_weights,
)

progress_bar = Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    TaskProgressColumn(),
    "Elapsed:",
    TimeElapsedColumn(),
    "Remaining:",
    TimeRemainingColumn(),
)


def train_loop_step(
    model, train_dataloader, device, optimizer, criterion, writer, epoch
):
    train_loss = 0
    model.train()
    epoch_train_running_loss = 0
    with progress_bar:
        task = progress_bar.add_task(
            description=f"[green]Train Epoch #{epoch+1}",
            total=len(train_dataloader),
        )
        for step, (noisy_images, _, target_images) in enumerate(train_dataloader):
            noisy_images = noisy_images.to(device)
            target_images = target_images.to(device)
            pred_images = model(noisy_images)
            train_loss = criterion(pred_images, target_images)
            train_loss.backward()
            if ((step + 1) % 2 == 0) or (step == (len(train_dataloader) - 1)):
                optimizer.step()
                optimizer.zero_grad()
            epoch_train_running_loss += train_loss.item()
            progress_bar.update(task, advance=1)
    epoch_train_loss_mean = epoch_train_running_loss / (step + 1)
    writer.add_scalar(
        "train loss",
        epoch_train_loss_mean,
        epoch,
    )
    return epoch_train_loss_mean


def test_loop_step(
    model,
    val_dataloader,
    device,
    criterion,
    writer,
    epoch,
):
    epoch_test_running_loss = 0
    epoch_ssim_running_score = 0
    epoch_psnr_running_score = 0
    with progress_bar:
        task = progress_bar.add_task(
            description=f"[blue]Train Epoch #{epoch+1}",
            total=len(val_dataloader),
        )
        with torch.no_grad():
            model.eval()
            for step, (noisy_images, _, target_images) in enumerate(val_dataloader):
                noisy_images = noisy_images.to(device)
                target_images = target_images.to(device)
                pred_images = model(noisy_images)
                val_loss = criterion(pred_images, target_images)
                epoch_test_running_loss += val_loss.item()
                epoch_ssim_running_score += SSIM(pred_images, target_images).item()
                epoch_psnr_running_score += PSNR(pred_images, target_images).item()
                progress_bar.update(task, advance=1)
    epoch_test_loss_mean = epoch_test_running_loss / (step + 1)
    epoch_ssim_score_mean = epoch_ssim_running_score / (step + 1)
    epoch_psnr_score_mean = epoch_psnr_running_score / (step + 1)
    return epoch_test_loss_mean, epoch_ssim_score_mean, epoch_psnr_score_mean


def validation_loop_step(
    model, device, tensor_board_dataset, writer, tensor_board_dataset_iterator, epoch
):
    with torch.no_grad():
        model.eval()
        if tensor_board_dataset:
            tb_noisy_images_array = []
            target_images_array = []
            tb_pred_images_array = []
            for _ in range(24):
                try:
                    tb_noisy_images, _, tb_target_images = next(
                        tensor_board_dataset_iterator
                    )
                except StopIteration:
                    tensor_board_dataset_iterator = iter(tensor_board_dataset)
                    tb_noisy_images, _, tb_target_images = next(
                        tensor_board_dataset_iterator
                    )
                tb_noisy_images = tb_noisy_images.to(device)
                target_images = tb_target_images.to(device)
                tb_pred_images = model(tb_noisy_images)
                tb_noisy_images_array.append(
                    tb_noisy_images.to("cpu").squeeze(0).clone()
                )
                target_images_array.append(target_images.to("cpu").squeeze(0).clone())
                tb_pred_images_array.append(tb_pred_images.to("cpu").squeeze(0).clone())
            log_images_to_tensorboard(
                writer,
                epoch,
                torch.stack(tb_noisy_images_array),
                torch.stack(target_images_array),
                torch.stack(tb_pred_images_array),
            )

    return tensor_board_dataset_iterator


def train_loop(
    model: SeConvDesnoiseNet,
    learning_rate: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    run_name: str,
    num_epochs: int = 100,
    tensor_board_dataset: Optional[DataLoader] = None,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MixL1SSIMLoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    tensor_board_dataset_iterator = iter(tensor_board_dataset)  # type: ignore
    for epoch in range(num_epochs):
        epoch_train_loss_mean = train_loop_step(
            model, train_dataloader, device, optimizer, criterion, writer, epoch
        )
        epoch_test_loss_mean, epoch_ssim_score_mean, epoch_psnr_score_mean = (
            test_loop_step(model, val_dataloader, device, criterion, writer, epoch)
        )
        log_progress_to_console(
            epoch_train_loss_mean,
            epoch_test_loss_mean,
            epoch,
            num_epochs,
            run_name,
            model,
        )
        log_test_to_tensor_board(
            writer,
            epoch,
            epoch_test_loss_mean,
            epoch_ssim_score_mean,
            epoch_psnr_score_mean,
        )
        tensor_board_dataset_iterator = validation_loop_step(
            model,
            device,
            tensor_board_dataset,
            writer,
            tensor_board_dataset_iterator,
            epoch,
        )
        save_model_weights(model, run_name, epoch)
