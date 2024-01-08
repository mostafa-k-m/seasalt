from typing import Optional

import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils import (
    log_images_to_tensorboard,
    log_progress_to_console,
    save_model_weights,
)
from .loss import DiceLoss
from .model import NoiseDetector

torch.manual_seed(101)


def train(
    model: NoiseDetector,
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, cooldown=5, min_lr=1e-5
    )
    criterion = DiceLoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        epoch_train_losses = []
        for step, (noisy_images, masks, _) in track(
            enumerate(train_dataloader),
            description=f"Train Epoch #{epoch+1}",
            total=len(train_dataloader),
        ):
            noisy_images = noisy_images.to(device)
            masks = masks.to(device)
            pred_masks = model(noisy_images)
            train_loss = criterion(pred_masks, masks)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
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
        with torch.no_grad():
            model.eval()
            for step, (noisy_images, masks, _) in track(
                enumerate(val_dataloader),
                description=f"Validation Epoch #{epoch+1}",
                total=len(val_dataloader),
            ):
                noisy_images = noisy_images.to(device)
                masks = masks.to(device)
                pred_masks = model(noisy_images)
                val_loss = criterion(pred_masks, masks)
                epoch_val_losses.append(val_loss)
            if tensor_board_dataset:
                tb_ix = torch.randint(0, len(tensor_board_dataset), (1, 1)).item()
                for ix, (tb_noisy_images, tb_masks, _) in enumerate(
                    tensor_board_dataset
                ):
                    if ix != tb_ix:
                        continue
                    tb_noisy_images = tb_noisy_images.to(device)
                    tb_masks = tb_masks.to(device)
                    tb_pred_masks = model(tb_noisy_images)
                    log_images_to_tensorboard(
                        writer,
                        epoch,
                        tb_noisy_images,
                        tb_masks,
                        tb_pred_masks,
                    )
        epoch_valid_loss_value = torch.mean(torch.stack(epoch_val_losses)).item()
        writer.add_scalar(
            "valid loss",
            epoch_valid_loss_value,
            epoch,
        )
        scheduler.step(epoch_valid_loss_value)
        log_progress_to_console(
            epoch_train_loss_value,
            epoch_valid_loss_value,
            epoch,
            num_epochs,
            run_name,
            model,
        )
        save_model_weights(model, run_name, epoch)
