import random

import numpy as np
import torch

from seasalt.salt_net import (
    HybridModel,
    NoiseType,
    get_tensor_board_dataset,
    get_test_dataloader,
    get_train_dataloader,
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


noise_type = NoiseType.RANDOM
min_noise = 0.5
max_noise = 0.95
batch_size = 18
train_dataloader = get_train_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)
val_dataloader = get_test_dataloader(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=batch_size
)
tb_dataloader = get_tensor_board_dataset(
    noise_type, min_noise=min_noise, max_noise=max_noise, batch_size=1
)

denoiser_model = HybridModel(
    denoiser_weights_path=None,
    detector_weights_path="./models/pytorch_noise_detector_prob_se_"
    "uneq_trick_low_lr_more_data_dropout_40.h5",
)

denoiser_model.to(device).compile()

train_hybrid_model(
    denoiser_model,  # type: ignore
    1e-3,
    train_dataloader,
    val_dataloader,
    device,
    "old_reliable_refined",
    100,
    tb_dataloader,
)
