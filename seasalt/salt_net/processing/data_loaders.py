import random
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .noise_adders import NoiseType, noise_adder

PATCH_SIZE, STRIDE = 64, 10
SCALES = [1, 0.9, 0.8, 0.7]
BATCH_SIZE = 128
torch.manual_seed(101)
np.random.seed(101)


root_folder = Path().resolve()

data_folder = root_folder.joinpath("data").joinpath("train").resolve()

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
    ]
)


def data_augmenter(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return torch.flip(img, [0])
    elif mode == 2:
        return torch.rot90(img, 1, [1, 2])
    elif mode == 3:
        return torch.flip(torch.rot90(img, 1, [1, 2]), [0])
    elif mode == 4:
        return torch.rot90(img, 2, [1, 2])
    elif mode == 5:
        return torch.flip(torch.rot90(img, 2, [1, 2]), [0])
    elif mode == 6:
        return torch.rot90(img, 3, [1, 2])
    elif mode == 7:
        return torch.flip(torch.rot90(img, 3, [1, 2]), [0])


def collate_images(
    noise_type, min_noise, max_noise, batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = [item[0] for item in batch]
    stacked_images = torch.stack(images)
    noisy_images, masks = noise_adder(
        stacked_images,
        (min_noise - max_noise) * torch.rand(stacked_images.shape) + max_noise,
        noise_type,
    )
    return noisy_images, masks, stacked_images


class AugmentedDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, n_patches_per_image=640):
        super(AugmentedDataset, self).__init__(root, transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        self.n_patches_per_image = n_patches_per_image
        self.samples = self.make_dataset(
            self.root, class_to_idx, datasets.folder.IMG_EXTENSIONS, is_valid_file=None
        )
        self.targets = [s[1] for s in self.samples]
        self.loaded_image_index = -1
        self.patches = {}

    def __len__(self) -> int:
        return len(self.samples) * self.n_patches_per_image

    def patch_generator(self, img) -> List[torch.Tensor]:
        h, w = img.shape[1:3]
        patches = []
        for s in SCALES:
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_scaled = tvF.resize(
                img,
                size=(w_scaled, h_scaled),  # type: ignore
                interpolation=tvF.InterpolationMode.BICUBIC,
                antialias=False,
            )
            for i in range(0, h_scaled - PATCH_SIZE + 1, STRIDE):
                for j in range(0, w_scaled - PATCH_SIZE + 1, STRIDE):
                    x = img_scaled[:, i : i + PATCH_SIZE, j : j + PATCH_SIZE]
                    patches.append(data_augmenter(x, mode=np.random.randint(0, 8)))
        return random.choices(patches, k=self.n_patches_per_image)

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx,
        extensions,
        is_valid_file,
    ):
        return datasets.folder.make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def patch_handler(self, index: int):
        if index not in self.patches:
            self.loaded_image_index = index
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            self.patches[index] = self.patch_generator(sample)
        return

    def __getitem__(self, index: int):
        self.patch_handler(index // self.n_patches_per_image)
        return (
            self.patches[index // self.n_patches_per_image][
                index % self.n_patches_per_image
            ],
            0,
        )


dataset = AugmentedDataset(root=str(data_folder), transform=transform)
tensor_board_dataset = datasets.ImageFolder(root=str(data_folder), transform=transform)
lengths = [round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
train_dataset, val_dataset = random_split(
    dataset, lengths, torch.Generator().manual_seed(101)
)


def get_train_dataloader(
    noise_type: NoiseType, min_noise: float, max_noise: float, batch_size: int
) -> DataLoader:
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_images,
            noise_type,
            min_noise,
            max_noise,
        ),
    )


def get_tensor_board_dataset(
    noise_type: NoiseType, min_noise: float, max_noise: float, batch_size: int
) -> DataLoader:
    return DataLoader(
        tensor_board_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_images,
            noise_type,
            min_noise,
            max_noise,
        ),
    )
