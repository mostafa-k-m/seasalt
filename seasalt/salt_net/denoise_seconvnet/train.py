import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils import log_images_to_tensorboard, log_progress_to_console
from .loss import GradientVariance
from .model import Desnoiser


def train(
    model: Desnoiser,
    learning_rate: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    run_name: str,
    num_epochs: int = 100,
    log_images: bool = False,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = GradientVariance(device)
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        epoch_train_losses = []
        for step, (noisy_images, masks, target_images) in track(
            enumerate(train_dataloader),
            description=f"train_epoch_#{epoch}",
            total=len(train_dataloader),
        ):
            noisy_images = noisy_images.to(device)
            noisy_images[masks == 1] = 0
            masks = masks.to(device)
            target_images = target_images.to(device)
            pred_images = model(noisy_images, masks)
            train_loss = criterion(
                pred_images, target_images, channels=noisy_images.shape[1]
            )
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_losses.append(train_loss)
            writer.add_scalar(
                "train loss", train_loss, epoch, len(train_dataloader) * epoch + step
            )

        epoch_val_losses = []
        with torch.no_grad():
            model.eval()
            for step, (noisy_images, masks, target_images) in enumerate(val_dataloader):
                noisy_images = noisy_images.to(device)
                noisy_images[masks == 1] = 0
                masks = masks.to(device)
                target_images = target_images.to(device)
                pred_images = model(noisy_images, masks)
                val_loss = criterion(
                    pred_images, target_images, channels=noisy_images.shape[1]
                )
                epoch_val_losses.append(val_loss)
                writer.add_scalar(
                    "valid loss", val_loss, epoch, len(val_dataloader) * epoch + step
                )
            if log_images:
                log_images_to_tensorboard(
                    writer,
                    epoch,
                    noisy_images,  # type: ignore
                    target_images,  # type: ignore
                    pred_images,  # type: ignore
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
