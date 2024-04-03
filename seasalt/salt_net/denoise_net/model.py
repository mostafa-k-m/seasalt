from typing import Tuple

import torch
import torch.nn.functional as F

from ..denoise_seconvnet import ConvBlock, OutputBlock, SeConvBlock
from ..noise_detector_unet import NoiseDetectorUNet as AutoEncoder


def reshape_4_to_3_dim(x):
    return x.reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])


def reshape_3_to_4_dim(x, h, w):
    return x.reshape(x.shape[0], x.shape[-1], h, w)


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, Tuple):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def normalize(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def forward(self, x):
        h, w = x.shape[-2:]
        return reshape_3_to_4_dim(self.normalize(reshape_4_to_3_dim(x)), h, w)


class DFFN(torch.nn.Module):
    def __init__(self, dim, chnl_expansion_factor):
        super(DFFN, self).__init__()

        hidden_features = int(dim * chnl_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = torch.nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=False
        )
        self.hidden_conv = torch.nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=False,
        )
        self.fft_params = torch.nn.Parameter(
            torch.ones(
                (hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)
            )
        )
        self.project_out = torch.nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.project_in(x)
        x_patch = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x_patch = x.reshape(
            x.shape[0],
            x.shape[1],
            int(x.shape[2] / self.patch_size),
            int(x.shape[3] / self.patch_size),
            self.patch_size,
            self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = torch.mul(x_patch_fft, self.fft_params.data)
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = x_patch.reshape(
            x_patch.shape[0],
            x_patch.shape[1],
            x_patch.shape[2] * self.patch_size,
            x_patch.shape[3] * self.patch_size,
        )
        x1, x2 = self.hidden_conv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(torch.nn.Module):
    def __init__(self, dim):
        super(FSAS, self).__init__()

        self.project_in = torch.nn.Conv2d(dim, dim * 6, kernel_size=1, bias=False)
        self.hidden_conv = torch.nn.Conv2d(
            dim * 6,
            dim * 6,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 6,
            bias=False,
        )

        self.project_out = torch.nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm = LayerNorm(dim * 2)
        self.patch_size = 8

    def forward(self, x):
        hidden = self.project_in(x)
        q, k, v = self.hidden_conv(hidden).chunk(3, dim=1)
        q_patch = q.reshape(
            q.shape[0],
            q.shape[1],
            int(q.shape[2] / self.patch_size),
            int(q.shape[3] / self.patch_size),
            self.patch_size,
            self.patch_size,
        )
        k_patch = k.reshape(
            k.shape[0],
            k.shape[1],
            int(k.shape[2] / self.patch_size),
            int(k.shape[3] / self.patch_size),
            self.patch_size,
            self.patch_size,
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = out.reshape(
            out.shape[0],
            out.shape[1],
            out.shape[2] * self.patch_size,
            out.shape[3] * self.patch_size,
        )
        output = v * out
        output = self.project_out(output)
        return output


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        chnl_expansion_factor,
    ):
        super(TransformerLayer, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = DFFN(dim, chnl_expansion_factor)

    def forward(self, x):
        x = x + self.ffn(self.norm(x))
        return x


class TransformerDownSampleBlock(torch.nn.Module):

    def __init__(
        self,
        dim,
        n_blocks,
        chnl_expansion_factor,
        downsample=True,
    ):
        super(TransformerDownSampleBlock, self).__init__()
        self.downsample = downsample
        self.transformer_block = torch.nn.Sequential(
            *[TransformerLayer(dim, chnl_expansion_factor) for _ in range(n_blocks)]
        )
        if self.downsample:
            self.downsample_block = torch.nn.Sequential(
                torch.nn.Upsample(
                    scale_factor=0.5, mode="bilinear", align_corners=False
                ),
                torch.nn.Conv2d(dim, dim * 2, 3, stride=1, padding=1),
            )

    def forward(self, x):
        x = self.transformer_block(x)
        if self.downsample:
            return self.downsample_block(x), x
        return x


class TransformerWithAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        chnl_expansion_factor,
    ):
        super(TransformerWithAttentionLayer, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = FSAS(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = DFFN(dim, chnl_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerUpSampleBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_blocks,
        chnl_expansion_factor,
        upsample=True,
    ):
        super(TransformerUpSampleBlock, self).__init__()
        self.upsample = upsample
        self.transformers = torch.nn.Sequential(
            *[
                TransformerWithAttentionLayer(dim, chnl_expansion_factor)
                for _ in range(n_blocks)
            ]
        )
        if self.upsample:
            self.upsample_block = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(dim, dim // 2, 3, stride=1, padding=1),
            )

    def forward(self, x):
        x = self.transformers(x)
        if self.upsample:
            x = self.upsample_block(x)
        return x


class FuseBlock(torch.nn.Module):

    def __init__(self, n_feat, chnl_expansion_factor):
        super(FuseBlock, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerDownSampleBlock(
            dim=n_feat * 2,
            n_blocks=1,
            downsample=False,
            chnl_expansion_factor=chnl_expansion_factor,
        )
        self.conv = torch.nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        return e + d


class FFTFormer(torch.nn.Module):
    def __init__(
        self,
        channels=1,
        dim=48,
        num_blocks=[6, 6, 12],
        num_refinement_blocks=4,
        chnl_expansion_factor=3,
    ):
        super(FFTFormer, self).__init__()

        self.patch_embed = torch.nn.Conv2d(
            channels, dim, kernel_size=3, stride=1, padding=1
        )
        self.encoders = torch.nn.ModuleList(
            [
                TransformerDownSampleBlock(
                    dim=dim * 2**ix,
                    n_blocks=n_blocks,
                    downsample=ix < len(num_blocks) - 1,
                    chnl_expansion_factor=chnl_expansion_factor,
                )
                for ix, n_blocks in enumerate(num_blocks)
            ]
        )
        self.middle_layer = TransformerUpSampleBlock(
            dim=dim * 2 ** (len(num_blocks) - 1),
            n_blocks=num_blocks[-1],
            upsample=True,
            chnl_expansion_factor=chnl_expansion_factor,
        )
        self.decoders = torch.nn.ModuleList(
            [
                TransformerUpSampleBlock(
                    dim=dim * 2**ix,
                    n_blocks=n_blocks,
                    upsample=ix != 0,
                    chnl_expansion_factor=chnl_expansion_factor,
                )
                for ix, n_blocks in reversed(list(enumerate(num_blocks)))
            ][1:]
        )
        self.refinement = TransformerUpSampleBlock(
            dim=dim,
            n_blocks=num_refinement_blocks,
            upsample=False,
            chnl_expansion_factor=chnl_expansion_factor,
        )

        self.fuse_layers = torch.nn.ModuleList(
            [
                FuseBlock(dim * 2**i, chnl_expansion_factor=chnl_expansion_factor)
                for i in range(len(num_blocks) - 1)
            ]
        )
        self.output = torch.nn.Conv2d(
            int(dim), channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, img):
        x = self.patch_embed(img.clone())
        encoder_outputs = []
        for encoder in self.encoders:
            if encoder.downsample:
                x, y = encoder(x)
                encoder_outputs.append(y)
            else:
                encoder_outputs.append(encoder(x))
        decoded = self.middle_layer(encoder_outputs[-1])
        for encoded, decoder, fuse in zip(
            reversed(encoder_outputs[: len(self.decoders)]),
            self.decoders,
            reversed(self.fuse_layers),
        ):
            decoded = decoder(fuse(decoded, encoded))
        refined = self.refinement(decoded)
        return self.output(refined) + img


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
        img[:, :, 1:, :] -= tv
        img[:, :, :-1, :] += tv
        th = self.gamma * ch * self.conv_dh(dh)
        img[:, :, :, 1:] -= th
        img[:, :, :, :-1] += th
        return img

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.conv(self.diffuse(img))


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
        enable_seconv=True,
        enable_anti_seconv=False,
        enable_unet=False,
        enable_fft=False,
        enable_anisotropic=True,
        enable_unet_post_processing=True,
    ) -> None:
        super(DenoiseNet, self).__init__()
        self.enable_seconv = enable_seconv
        self.enable_anti_seconv = enable_anti_seconv
        self.enable_unet = enable_unet
        self.enable_fft = enable_fft
        self.enable_anisotropic = enable_anisotropic
        self.enable_unet_post_processing = enable_unet_post_processing

        n_outputs = 1

        if self.enable_unet:
            n_outputs += 1
            self.auto_encoder = AutoEncoder(
                channels, auto_encoder_first_output, auto_encoder_depth
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
            self.unet_post_processing = FFTFormer(channels)

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
        if self.enable_anti_seconv:
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
        if self.enable_anisotropic:
            x_anisotropic = noisy_images.clone()
            for module in self.anisotropic_blocks:
                x_anisotropic = module(x_anisotropic)
            outputs.append(x_anisotropic)
        output = torch.cat(outputs, 1)
        for module in self.output_layer:
            output = module(output)
        if self.enable_unet_post_processing:
            return self.unet_post_processing(output)
        return output
