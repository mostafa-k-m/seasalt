from pathlib import Path

import numpy as np
import torch

from seasalt.salt_net import (
    HybridModel,
)

root_folder = Path(__file__).parent.parent.parent.resolve()
models_folder = root_folder.joinpath("models")
denoiser = torch.jit.load(models_folder.joinpath("denoiser.pt"))


class SaltNetOneStageHandler:
    def __init__(self, denoiser_path="hybrid.pt"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        # self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # self.denoiser = torch.jit.load(models_folder.joinpath(denoiser_path))
        self.denoiser = HybridModel(
            denoiser_weights_path=None,
            detector_weights_path=None,
        )
        self.denoiser.to(self.device)
        self.denoiser.load_state_dict(
            torch.load(denoiser_path, self.device),
        )
        self.denoiser.to(self.device)
        self.denoiser = self.denoiser.eval().float()

        self.initialized = True

    def preprocess(self, data: np.ndarray):
        data = data.astype(np.float64)
        if len(data.shape) == 2:
            data = data.reshape(1, *data.shape)
        if np.max(data) > 1:
            data /= 255
        cropped_array = data[
            :,
            : data.shape[-2] - (data.shape[-2] % 2),
            : data.shape[-1] - (data.shape[-1] % 2),
        ]
        X = torch.tensor(cropped_array).unsqueeze(0).float().to(self.device)
        return X

    def inference(self, inputs):
        with torch.no_grad():
            return self.denoiser(inputs)

    def predict(self, data):
        return (
            self.inference(self.preprocess(data)).squeeze().squeeze().detach().numpy()
            * 255
        )


class SaltNetTwoStageHandler(SaltNetOneStageHandler):
    def __init__(self, denoiser_path="denoiser.pt", detector_path="detector.pt"):
        super().__init__(denoiser_path)
        self.detector = torch.jit.load(models_folder.joinpath(detector_path))
        self.detector.to(self.device)
        self.detector = self.detector.eval().float()

    def inference(self, inputs):
        mask = self.detector(inputs.float())
        return self.denoiser(
            inputs,
            torch.logical_or(torch.round(mask), (inputs == 1) | (inputs == 0)).float(),
        )
