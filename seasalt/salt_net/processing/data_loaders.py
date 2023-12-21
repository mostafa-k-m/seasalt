from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .noise_adders import NoiseType, noise_adder

root_folder = Path().resolve()

data_folder = root_folder.joinpath("data").joinpath("train").resolve()

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
    ]
)
torch.manual_seed(101)


def collate_images(
    noise_type, min_noise, max_noise, batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = [item[0] for item in batch]
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)
    padded_images = [
        F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]))
        for img in images
    ]
    stacked_images = torch.stack(padded_images)
    noisy_images, masks = noise_adder(
        stacked_images,
        (min_noise - max_noise) * torch.rand(stacked_images.shape) + max_noise,
        noise_type,
    )
    return noisy_images, masks, stacked_images


dataset = datasets.ImageFolder(root=str(data_folder), transform=transform)
lengths = [round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
train_dataset, val_dataset = random_split(dataset, lengths)


def get_train_dataloader(
    noise_type: NoiseType, min_noise: float, max_noise: float, batch_size: int
) -> DataLoader:
    return DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(
            collate_images,
            noise_type,
            min_noise,
            max_noise,
        ),
    )


def get_test_dataloader(
    noise_type: NoiseType, min_noise: float, max_noise: float, batch_size: int
) -> DataLoader:
    return DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(
            collate_images,
            noise_type,
            min_noise,
            max_noise,
        ),
    )
