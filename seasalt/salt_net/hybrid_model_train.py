from typing import Optional

import torch
import torch.optim as optim
from .denoise_seconvnet.loss import MixL1SSIMLoss
from .denoise_seconvnet.model import Desnoiser
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from .utils import (
    PSNR,
    SSIM,
    log_images_to_tensorboard,
    log_progress_to_console,
    log_validation_to_tensor_board,
    save_model_weights,
)


def train(
    model: Desnoiser,
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
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=5, cooldown=5, min_lr=1e-5
    # )
    criterion = MixL1SSIMLoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        epoch_train_losses = []
        for step, (noisy_images, _, target_images) in track(
            enumerate(train_dataloader),
            description=f"Train Epoch #{epoch+1}",
            total=len(train_dataloader),
        ):
            noisy_images = noisy_images.to(device)
            target_images = target_images.to(device)
            pred_images = model(noisy_images)
            train_loss = criterion(pred_images, target_images)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_losses.append(train_loss)
        epoch_train_loss_value = torch.mean(torch.stack(epoch_train_losses)).item()
        writer.add_scalar(
            "train loss",
            epoch_train_loss_value,
            epoch,
        )
        epoch_val_losses = []
        epoch_ssim_scores = []
        epoch_psnr_scores = []
        with torch.no_grad():
            model.eval()
            for step, (noisy_images, _, target_images) in track(
                enumerate(val_dataloader),
                description=f"Validation Epoch #{epoch+1}",
                total=len(val_dataloader),
            ):
                noisy_images = noisy_images.to(device)
                target_images = target_images.to(device)
                pred_images = model(noisy_images)
                val_loss = criterion(pred_images, target_images)
                epoch_val_losses.append(val_loss)
                epoch_ssim_scores.append(SSIM(pred_images, target_images))
                epoch_psnr_scores.append(PSNR(pred_images, target_images))
            if tensor_board_dataset:
                tb_ix = torch.randint(0, len(tensor_board_dataset), (1, 1)).item()
                for ix, (tb_noisy_images, _, tb_target_images) in enumerate(
                    tensor_board_dataset
                ):
                    if ix != tb_ix:
                        continue
                    tb_noisy_images = tb_noisy_images.to(device)
                    target_images = tb_target_images.to(device)
                    tb_pred_images = model(tb_noisy_images)
                    log_images_to_tensorboard(
                        writer,
                        epoch,
                        tb_noisy_images,
                        tb_target_images,
                        tb_pred_images,
                    )
                    break
        epoch_valid_loss_value = torch.mean(torch.stack(epoch_val_losses)).item()
        epoch_ssim_score = torch.mean(torch.stack(epoch_ssim_scores)).item()
        epoch_psnr_score = torch.mean(torch.stack(epoch_psnr_scores)).item()
        log_validation_to_tensor_board(
            writer,
            epoch,
            epoch_valid_loss_value,
            epoch_ssim_score,
            epoch_psnr_score,
        )
        # scheduler.step(epoch_valid_loss_value)
        log_progress_to_console(
            epoch_train_loss_value,
            epoch_valid_loss_value,
            epoch,
            num_epochs,
            run_name,
            model,
        )
        save_model_weights(model, run_name, epoch)
