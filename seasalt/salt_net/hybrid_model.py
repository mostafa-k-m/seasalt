import torch

from .denoise_net import DenoiseNet
from .denoise_net.model import AutoEncoder as NoiseDetectorUNet

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class HybridModel(torch.nn.Module):
    def __init__(self, denoiser_weights_path, detector_weights_path):
        super(HybridModel, self).__init__()
        denoiser_model = DenoiseNet(
            output_cnn_depth=10,
            enable_seconv=True,
            enable_anti_seconv=False,
            enable_unet=False,
            enable_fft=False,
            enable_anisotropic=True,
            enable_unet_post_processing=True,
        )
        if denoiser_weights_path:
            denoiser_model.load_state_dict(
                torch.load(denoiser_weights_path, device),
            )
        noise_detecor_model = NoiseDetectorUNet(squeeze_excitation=True, dropout=True)
        if detector_weights_path:
            noise_detecor_model.load_state_dict(
                torch.load(detector_weights_path, device),
            )

        self.denoiser = denoiser_model.to(device)
        self.detecor = noise_detecor_model.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        masks = self.detecor(images)
        return self.denoiser(
            images,
            torch.round(masks),
        )
