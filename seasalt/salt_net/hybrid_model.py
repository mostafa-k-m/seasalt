import torch

from .denoise_net import DenoiseNet
from .denoise_net.model import AutoEncoder as AutoEncoder

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class HybridModel(torch.nn.Module):
    def __init__(
        self,
        denoiser_weights_path,
        enable_seconv=True,
        enable_fft=False,
        enable_anisotropic=True,
        enable_unet_post_processing=True,
        transformer_depth=10,
    ):
        super(HybridModel, self).__init__()
        denoiser_model = DenoiseNet(
            enable_seconv=enable_seconv,
            enable_fft=enable_fft,
            enable_anisotropic=enable_anisotropic,
            enable_unet_post_processing=enable_unet_post_processing,
            transformer_depth=transformer_depth,
        )
        if denoiser_weights_path:
            denoiser_model.load_state_dict(
                torch.load(denoiser_weights_path, device),
            )
        self.denoiser = denoiser_model.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.denoiser(images)
