import random

import numpy as np
import torch

from seasalt.salt_net import (
    DataLoadersInitializer,
    HybridModel,
    NoiseType,
    train_hybrid_model,
)

torch.manual_seed(101)
np.random.seed(101)
random.seed(101)

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
# device = torch.device("mps")
else:
    device = torch.device("cpu")

dli = DataLoadersInitializer()

noise_type = NoiseType.RANDOM
min_noise = 0.5
max_noise = 0.95
batch_size = 20

train_dataloader = dli.get_train_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)
val_dataloader = dli.get_test_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)

tb_dataloader = dli.get_tensor_board_dataset(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=1
)
torch.set_float32_matmul_precision("high")

denoiser_model = HybridModel(
    denoiser_weights_path=None,
    enable_seconv=False,
    enable_fft=False,
    enable_anisotropic=True,
    enable_unet_post_processing=True,
    transformer_depth=10,
)
denoiser_model.to(device)  # .compile()
train_hybrid_model(
    denoiser_model,  # type: ignore
    1e-3,
    train_dataloader,
    val_dataloader,
    device,
    "ablation_no_seconv",
    250,
    tb_dataloader,
)
