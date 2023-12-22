import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils import log_images_to_tensorboard, log_progress_to_console
from .loss import DiceLoss
from .model import NoiseDetector


def train(
    model: NoiseDetector,
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = DiceLoss()
    writer = SummaryWriter(log_dir=f".runs/{run_name}")
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        epoch_train_losses = []
        for step, (noisy_images, masks, _) in track(
            enumerate(train_dataloader),
            description=f"Train Epoch #{epoch}",
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
            writer.add_scalar(
                "train loss", train_loss, epoch, len(train_dataloader) * epoch + step
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
                writer.add_scalar(
                    "valid loss", val_loss, epoch, len(val_dataloader) * epoch + step
                )
            if log_images:
                log_images_to_tensorboard(
                    writer,
                    epoch,
                    noisy_images,  # type: ignore
                    masks,  # type: ignore
                    pred_masks,  # type: ignore
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
