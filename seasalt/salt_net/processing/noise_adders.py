from enum import Enum
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch


class NoiseType(Enum):
    GUASSIAN = 1
    SAP = 2
    BERNOULLI = 3
    POISSON = 4
    RANDOM = 5

    def get_noise_func(
        self,
    ) -> Callable:
        funcs: Dict[str, List[Callable]] = {
            "GUASSIAN": [_gaussian_noise_adder],
            "SAP": [_salt_and_pepper_noise_adder],
            "BERNOULLI": [_bernoulli_noise_adder],
            "POISSON": [_poisson_noise_adder],
            "RANDOM": [
                _gaussian_noise_adder,
                _salt_and_pepper_noise_adder,
                _bernoulli_noise_adder,
                _poisson_noise_adder,
            ],
        }
        return np.random.choice(funcs[self.name])  # type: ignore


def noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor, noise_type: NoiseType
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_func = noise_type.get_noise_func()
    noisy_images, masks = noise_func(images, noise_parameters)
    return noisy_images, masks.float()


def _gaussian_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.stack(
        [
            torch.normal(0, std.item(), images.shape[1:])
            for std in noise_parameters[:, 0, 0, 0]
        ]
    )
    noisy_images = (images + noise).clip(0, 1)
    return (
        noisy_images,
        torch.logical_and(
            torch.logical_or(noisy_images == 0, noisy_images == 1),
            ~torch.logical_or(images == 0, images == 1),
        )
        # torch.logical_or(
        #     noise > noise[noise >= 0].quantile(0.1),
        #     noise < noise[noise < 0].quantile(0.1),
        # ),
    )


def _salt_and_pepper_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    salt_mask = torch.rand(images.shape) < (noise_parameters / 2)
    images = images.masked_fill(salt_mask, 1.0)
    pepper_mask = torch.rand(images.shape) < (noise_parameters / 2)
    images = images.masked_fill(pepper_mask, 0.0)
    return images, torch.logical_or(salt_mask, pepper_mask)


def _bernoulli_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = noise_parameters * torch.ones(images.shape)
    noise = torch.bernoulli(a)
    noisy_images = images * noise
    return noisy_images, noise == 0


def _poisson_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = noise_parameters * torch.ones(images.shape)
    p = torch.poisson(a)
    noise = p / p.max()
    noisy_images = (images + noise).clip(0, 1)
    return noisy_images, noise > 0
