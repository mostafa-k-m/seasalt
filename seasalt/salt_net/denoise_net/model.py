import math

import torch
import torch.nn.functional as F

from ..denoise_seconvnet import ConvBlock, OutputBlock, SeConvBlock
from ..noise_detector_unet import AutoEncoder


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


class CalculateDiff(torch.nn.Module):
    def __init__(self, kernel):
        super(CalculateDiff, self).__init__()
        self.kernel = torch.nn.Parameter(
            torch.tensor(kernel, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x) -> torch.Tensor:
        return F.conv2d(x, self.kernel)


class AnisotropicDiffusionBlock(torch.nn.Module):

    def __init__(self, channels, kernel_size, gamma: float = 0.25, kappa: float = 100):
        super(AnisotropicDiffusionBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.diff_v = CalculateDiff([[[[-1], [1]]]])
        self.diff_h = CalculateDiff([[[[-1, 1]]]])
        self.gamma = torch.nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.kappa = torch.nn.Parameter(torch.tensor(kappa, dtype=torch.float32))
        self.conv_dv = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )
        self.conv_dh = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )
        self.conv = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + (torch.abs((x * x)) / (self.kappa * self.kappa)))

    def c(
        self, dv: torch.Tensor, dh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cv = self.g(torch.mean(torch.abs(dv), 1, keepdim=True))
        ch = self.g(torch.mean(torch.abs(dh), 1, keepdim=True))
        return cv, ch

    def diffuse(self, img: torch.Tensor) -> torch.Tensor:
        dv = self.diff_v(img)
        dh = self.diff_h(img)
        cv, ch = self.c(dv, dh)
        tv = self.gamma * cv * self.conv_dv(dv)
        img = img - torch.nn.functional.pad(tv, (0, 0, 1, 0))
        img = img + torch.nn.functional.pad(tv, (0, 0, 0, 1))
        th = self.gamma * ch * self.conv_dh(dh)
        img = img - torch.nn.functional.pad(th, (1, 0, 0, 0))
        img = img + torch.nn.functional.pad(th, (0, 1, 0, 0))
        return img

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.conv(self.diffuse(img))


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones((normalized_shape,)))
        self.bias = torch.nn.Parameter(torch.zeros((normalized_shape,)))

    def normalize(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, h * w, c)  # reshape to 3 dims
        x = self.normalize(x)
        return x.reshape(b, c, h, w)  # reshape to 4 dims


class MDTA(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = torch.nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = torch.nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = torch.nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=False,
        )
        self.project_out = torch.nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        chnls_per_head = c // self.num_heads
        q = q.reshape(b, self.num_heads, chnls_per_head, h * w)
        k = k.reshape(b, self.num_heads, chnls_per_head, h * w)
        v = v.reshape(b, self.num_heads, chnls_per_head, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, self.num_heads * chnls_per_head, h, w)
        return self.project_out(out)


class GDFN(torch.nn.Module):
    def __init__(self, dim, chnl_expansion_factor):
        super(GDFN, self).__init__()

        hidden_features = int(dim * chnl_expansion_factor)
        self.project_in = torch.nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=False
        )
        self.dconv = torch.nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=False,
        )
        self.project_out = torch.nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerLayer(torch.nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        chnl_expansion_factor,
    ):
        super(TransformerLayer, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, chnl_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        channels,
        hidden_dim,
        n_blocks,
        num_heads,
        chnl_expansion_factor,
    ):
        super(TransformerBlock, self).__init__()
        self.transformers = torch.nn.Sequential(
            *[
                TransformerLayer(hidden_dim, num_heads, chnl_expansion_factor)
                for _ in range(n_blocks)
            ]
        )
        self.output = torch.nn.Conv2d(
            hidden_dim, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.transformers(x)
        return self.output(x)


class DenoiseNet(torch.nn.Module):

    def __init__(
        self,
        channels=1,
        auto_encoder_first_output=64,
        auto_encoder_depth=5,
        seconv_depth=7,
        fft_depth=7,
        anisotropi_depth=5,
        output_cnn_depth=10,
        max_filters=64,
        refinement_transformer_n_blocks=4,
        refinement_transformer_num_heads=1,
        refinement_transformer_chnl_expansion_factor=2,
        enable_seconv=True,
        enable_anti_seconv=False,
        enable_unet=False,
        enable_fft=False,
        enable_anisotropic=True,
        enable_unet_post_processing=True,
        enable_refinement_transformer=True,
    ) -> None:
        super(DenoiseNet, self).__init__()
        self.enable_seconv = enable_seconv
        self.enable_anti_seconv = enable_anti_seconv
        self.enable_unet = enable_unet
        self.enable_fft = enable_fft
        self.enable_anisotropic = enable_anisotropic
        self.enable_unet_post_processing = enable_unet_post_processing
        self.enable_refinement_transformer = enable_refinement_transformer

        n_outputs = 1

        if self.enable_unet:
            n_outputs += 1
            self.auto_encoder = AutoEncoder(
                channels,
                auto_encoder_first_output,
                auto_encoder_depth,
            )

        if self.enable_seconv:
            n_outputs += 1
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
        if self.enable_anti_seconv:
            n_outputs += 1
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

        if self.enable_anisotropic:
            n_outputs += 1
            self.anisotropic_blocks = torch.nn.ModuleList(
                [
                    AnisotropicDiffusionBlock(channels=channels, kernel_size=3 + 2 * d)
                    for d in range(anisotropi_depth)
                ]
            )

        start_ix = 0
        if 2 ** (output_cnn_depth - 1) > n_outputs * channels:
            for start_ix in range(output_cnn_depth):
                if 2**start_ix > n_outputs * channels:
                    break
        else:
            start_ix = math.ceil(math.log2(n_outputs * channels))

        self.cnn_layers = torch.nn.ModuleList(
            [ConvBlock(n_outputs * channels, min(2**start_ix, max_filters))]
            + [
                ConvBlock(
                    min(2 ** (p + start_ix), max_filters),
                    (
                        max_filters
                        if p == output_cnn_depth - 2
                        else min(2 ** (p + start_ix + 1), max_filters)
                    ),
                )
                for p in range(output_cnn_depth - 1)
            ]
        )
        self.output_cnn_layer = ConvBlock(
            min(2 ** (output_cnn_depth - 2), max_filters), channels
        )

        if self.enable_unet_post_processing:
            self.unet_post_processing = AutoEncoder(
                channels,
                auto_encoder_first_output,
                auto_encoder_depth,
            )

        if self.enable_refinement_transformer:
            self.refinement_block = TransformerBlock(
                channels=channels,
                hidden_dim=max_filters,
                n_blocks=refinement_transformer_n_blocks,
                num_heads=refinement_transformer_num_heads,
                chnl_expansion_factor=refinement_transformer_chnl_expansion_factor,
            )

    def forward(self, noisy_images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = [noisy_images]
        if self.enable_unet:
            x_unet = noisy_images.clone()
            x_unet = self.auto_encoder(x_unet)
            outputs.append(x_unet)
        if self.enable_seconv:
            og_noisy_img = noisy_images.clone()
            og_noisy_img[mask == 1] = 0
            x_seconv = og_noisy_img.clone()
            for module in self.seconv_blocks:
                x_seconv = module(x_seconv, mask)
            for ix, module in enumerate(self.seconv_post_processing):
                if ix < len(self.seconv_post_processing) - 1:
                    x_seconv = module(x_seconv)
                else:
                    x_seconv = module(og_noisy_img, x_seconv, mask)
            outputs.append(x_seconv)
        if self.enable_anti_seconv:
            anti_mask = (~mask.bool()).float()
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
        if self.enable_anisotropic:
            x_anisotropic = noisy_images.clone()
            for module in self.anisotropic_blocks:
                x_anisotropic = module(x_anisotropic)
            outputs.append(x_anisotropic)
        embeddings = torch.cat(outputs, 1)
        for module in self.cnn_layers:
            embeddings = module(embeddings)
        output = self.output_cnn_layer(embeddings)
        if self.enable_refinement_transformer:
            output = output + self.refinement_block(embeddings)
        if self.enable_unet_post_processing:
            output = self.unet_post_processing(output)
        return output
