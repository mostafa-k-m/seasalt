import torch

from ..denoise_seconvnet import ConvBlock, OutputBlock, SeConvBlock
from ..noise_detector_unet import NoiseDetectorUNet as AutoEncoder


class FFTBlock(torch.nn.Module):
    def __init__(self, channels=1, filters=64):
        super(FFTBlock, self).__init__()

        self.ftt_conv = torch.nn.ModuleList(
            [
                ConvBlock(2, filters),
                ConvBlock(filters, filters),
                ConvBlock(filters, 2),
            ],
        )

        self.conv = torch.nn.ModuleList(
            [
                ConvBlock(channels, filters),
                ConvBlock(filters, filters),
                ConvBlock(filters, channels),
            ],
        )
        self.output_conv = ConvBlock(3 * channels, channels)
        self.output_relu = torch.nn.ReLU()

    def forward(self, noisy_images):
        fftshifted = torch.fft.fftshift(torch.fft.rfftn(noisy_images, dim=(-1, -2)))
        x_1 = torch.cat((fftshifted.real, fftshifted.imag), 1)
        for layer in self.ftt_conv:
            x_1 = layer(x_1)
        x_2 = noisy_images.clone()
        for layer in self.conv:
            x_2 = layer(x_2)
        x_1 = torch.fft.irfftn(
            torch.fft.ifftshift(torch.complex(x_1[:, :1, :, :], x_1[:, 1:2, :, :])),
            dim=(-1, -2),
        ).float()
        return self.output_relu(
            self.output_conv(torch.cat((x_1, x_2, noisy_images), 1))
        )


class DenoiseNet(torch.nn.Module):
    def __init__(
        self,
        channels=1,
        auto_encoder_first_output=64,
        auto_encoder_depth=5,
        seconv_depth=7,
        fft_depth=7,
        output_cnn_depth=20,
        max_filters=64,
        enable_seconv=True,
        enable_unet=False,
        enable_fft=True,
        enable_unet_post_processing=True,
    ) -> None:
        super(DenoiseNet, self).__init__()
        self.enable_seconv = enable_seconv
        self.enable_unet = enable_unet
        self.enable_fft = enable_fft
        self.enable_unet_post_processing = enable_unet_post_processing

        n_outputs = 1

        if self.enable_unet:
            n_outputs += 1
            self.auto_encoder = AutoEncoder(
                channels, auto_encoder_first_output, auto_encoder_depth
            )

        if self.enable_seconv:
            n_outputs += 2
            self.seconv_blocks = torch.nn.ModuleList(
                [
                    SeConvBlock(kernel_size=7 + 2 * d, channels=channels)
                    for d in range(seconv_depth)
                ]
            )
            self.seconv_post_processing = torch.nn.ModuleList(
                [
                    ConvBlock(channels, channels),
                    OutputBlock(channels, channels),
                ]
            )
            self.anti_seconv_blocks = torch.nn.ModuleList(
                [
                    SeConvBlock(kernel_size=7 + 2 * d, channels=channels)
                    for d in range(seconv_depth)
                ]
            )
            self.anti_seconv_post_processing = torch.nn.ModuleList(
                [
                    ConvBlock(channels, channels),
                    OutputBlock(channels, channels),
                ]
            )

        if self.enable_fft:
            n_outputs += 1
            self.fft_blocks = torch.nn.ModuleList(
                [FFTBlock(channels, max_filters) for _ in range(fft_depth)]
            )

        start_ix = 0
        for start_ix in range(output_cnn_depth):
            if 2**start_ix > n_outputs * channels:
                break

        self.output_layer = torch.nn.ModuleList(
            [ConvBlock(n_outputs * channels, 2**start_ix)]
            + [
                ConvBlock(
                    min(2 ** (p + start_ix), max_filters),
                    min(2 ** (p + start_ix + 1), max_filters),
                )
                for p in range(output_cnn_depth - 1)
            ]
            + [ConvBlock(min(2 ** (output_cnn_depth - 1), max_filters), channels)]
        )

        if self.enable_unet_post_processing:
            self.unet_post_processing = AutoEncoder(
                channels, auto_encoder_first_output, auto_encoder_depth
            )

    def forward(self, noisy_images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = [noisy_images]
        if self.enable_unet:
            x_unet = noisy_images.clone()
            x_unet = self.auto_encoder(x_unet)
            outputs.append(x_unet)
        if self.enable_seconv:
            og_mask = mask.clone()
            og_noisy_img = noisy_images.clone()
            og_noisy_img[og_mask == 1] = 0
            x_seconv = og_noisy_img.clone()
            for module in self.seconv_blocks:
                x_seconv = module(x_seconv, og_mask)
            for ix, module in enumerate(self.seconv_post_processing):
                if ix < len(self.seconv_post_processing) - 1:
                    x_seconv = module(x_seconv)
                else:
                    x_seconv = module(og_noisy_img, x_seconv, og_mask)
            outputs.append(x_seconv)
            anti_mask = (~mask.bool()).clone().float()
            anti_noisy_img = noisy_images.clone()
            anti_noisy_img[anti_mask == 1] = 0
            x_anti_seconv = anti_noisy_img.clone()
            for module in self.anti_seconv_blocks:
                x_anti_seconv = module(x_anti_seconv, anti_mask)
            for ix, module in enumerate(self.anti_seconv_post_processing):
                if ix < len(self.anti_seconv_post_processing) - 1:
                    x_anti_seconv = module(x_anti_seconv)
                else:
                    x_anti_seconv = module(anti_noisy_img, x_anti_seconv, anti_mask)
            outputs.append(x_anti_seconv)
        if self.enable_fft:
            x_fft = noisy_images.clone()
            for module in self.fft_blocks:
                x_fft = module(x_fft)
            outputs.append(x_fft)
        output = torch.cat(outputs, 1)
        for module in self.output_layer:
            output = module(output)
        if self.enable_unet_post_processing:
            return self.unet_post_processing(output)
        return output
