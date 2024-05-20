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
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dli = DataLoadersInitializer()

noise_type = NoiseType.RANDOM
min_noise = 0.5
max_noise = 0.95
batch_size = 1

train_dataloader = dli.get_train_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)
val_dataloader = dli.get_test_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)

tb_dataloader = dli.get_tensor_board_dataset(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=1
)

denoiser_model = HybridModel(denoiser_weights_path=None)
denoiser_model.to(device)  # .compile()

denoiser_model.load_state_dict(
    torch.load("./models/pytorch_transformers_instead_of_cnn_128_l1_10.h5", device),
)
try:
    train_hybrid_model(
        denoiser_model,  # type: ignore
        1e-4,
        train_dataloader,
        val_dataloader,
        device,
        "transformers_instead_of_cnn_256_l1",
        100,
        tb_dataloader,
    )
except Exception as e:  # noqa
    dli.destroy_tmp_dir()
    raise KeyboardInterrupt from e

dli.destroy_tmp_dir()
