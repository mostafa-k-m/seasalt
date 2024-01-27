from enum import Enum
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

torch.manual_seed(101)
np.random.seed(101)


class NoiseType(Enum):
    GUASSIAN = 1
    SAP = 2
    BERNOULLI = 3
    POISSON = 4
    RANDOM = 5
    PROBALISTIC = 6

    def get_noise_func(self, n_images) -> List[Callable]:
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
            "PROBALISTIC": [
                _gaussian_noise_adder,
                _poisson_noise_adder,
            ],
        }
        return np.random.choice(funcs[self.name], n_images).tolist()  # type: ignore


def noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor, noise_type: NoiseType
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_funcs = noise_type.get_noise_func(images.shape[0])
    noisy_images = []
    masks = []
    for i in range(images.shape[0]):
        noisy_image, mask = noise_funcs[i](
            images[i : i + 1], noise_parameters[i : i + 1, :, :, :]
        )
        mask = torch.logical_and(
            mask, torch.abs(noisy_image - images[i : i + 1]) >= 0.05
        )
        noisy_images.append(noisy_image)
        masks.append(mask.float())
    return torch.stack(noisy_images).view(*images.shape), torch.stack(masks).view(
        *images.shape
    )


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
        torch.abs(noise)
        > noise_parameters[:, 0, 0, 0]
        .view(-1, 1, 1, 1)
        .expand(-1, images.shape[1], images.shape[2], images.shape[3]),
    )


def _salt_and_pepper_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_parameters = noise_parameters[:, :1, :1, :1]
    randomness_mask = torch.rand(images.shape)
    salt_mask = randomness_mask < (noise_parameters / 2)
    images = images.masked_fill(salt_mask, 1.0)
    pepper_mask = torch.logical_and(
        randomness_mask >= (noise_parameters / 2),
        randomness_mask < noise_parameters,
    )
    images = images.masked_fill(pepper_mask, 0.0)
    return images, torch.logical_or(salt_mask, pepper_mask)


def _bernoulli_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = (
        noise_parameters[:, 0, 0, 0]
        .view(-1, 1, 1, 1)
        .expand(-1, images.shape[1], images.shape[2], images.shape[3])
    )
    noise = torch.bernoulli(a)
    noisy_images = images * noise
    return noisy_images, noise == 0


def _poisson_noise_adder(
    images: torch.Tensor, noise_parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = (
        noise_parameters[:, 0, 0, 0]
        .view(-1, 1, 1, 1)
        .expand(-1, images.shape[1], images.shape[2], images.shape[3])
    )
    p = torch.poisson(a)
    noise = p / p.max()
    noisy_images = (images + noise).clip(0, 1)
    return (
        noisy_images,
        noise
        > (
            torch.mean(noise, axis=(-1, -2, -3))  # type: ignore
            .view(-1, 1, 1, 1)
            .expand(-1, images.shape[1], images.shape[2], images.shape[3])
        ),
    )
