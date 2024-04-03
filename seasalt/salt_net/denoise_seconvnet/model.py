from itertools import chain
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x):
    return x.reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])


def to_4d(x, h, w):
    return x.reshape(x.shape[0], x.shape[-1], h, w)


class SeConvBlock(torch.nn.Module):
    def __init__(self, kernel_size, channels):
        super(SeConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        self.image_conv = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )

        self.M_hat_conv = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )

        self.R_conv = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
            groups=self.channels,
        )

    def forward(self, images, noise_mask):
        # non-noisy pixels map:
        M = noise_mask
        M_hat = 1 - M

        conv_input = self.image_conv(images)
        conv_M_hat = self.M_hat_conv(M_hat)

        # find 0 in conv_M_hat and change to 1:
        change_zero_to_one_conv_M_hat = conv_M_hat + (conv_M_hat == 0).float()

        S = conv_input / change_zero_to_one_conv_M_hat

        R = self.R_conv(M_hat)

        R = R >= self.kernel_size - 2
        R = R.float()

        y = S * R * M + images

        return y


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",  # Assuming "same" padding
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class OutputBlock(torch.nn.Module):
    def __init__(self, filters, channels, kernel_size=3, stride=1):
        super(OutputBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                filters,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",  # Assuming "same" padding
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels, momentum=0.1, eps=1e-5),
            torch.nn.ReLU(),
        )

    def forward(self, noisy_images, seconv_output, mask):
        x = self.conv(seconv_output)
        x = x * mask
        outputs = x + noisy_images
        return outputs


class SeConvDesnoiseNet(torch.nn.Module):
    def __init__(
        self, channels=1, seconv_depth=5, conv_depth=10, max_filters=128
    ) -> None:
        super(SeConvDesnoiseNet, self).__init__()
        self.seconv_blocks = torch.nn.ModuleList(
            [
                SeConvBlock(kernel_size=7 + 2 * d, channels=channels)
                for d in range(seconv_depth)
            ]
        )

        self.conv_layers = torch.nn.ModuleList(
            [ConvBlock(channels, 4)]
            + [
                ConvBlock(
                    min(2 ** (p + 2), max_filters), min(2 ** (p + 3), max_filters)
                )
                for p in range(conv_depth - 1)
            ]
        )

        self.output = OutputBlock(min(2 ** (conv_depth + 1), max_filters), channels)

    def forward(self, noisy_images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        noisy_clone = noisy_images.clone()
        noisy_clone[mask == 1] = 0
        x = noisy_clone

        for module in self.seconv_blocks:
            x = module(x, mask)

        for i, module in enumerate(self.conv_layers):
            x = module(x)

        return self.output(noisy_clone, x, mask)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, Tuple):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def normalize(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.normalize(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
        )
        self.fft = nn.Parameter(
            torch.ones(
                (hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)
            )
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

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
        x_patch_fft = torch.mul(x_patch_fft, self.fft.data)
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = x_patch.reshape(
            x_patch.shape[0],
            x_patch.shape[1],
            x_patch.shape[2] * self.patch_size,
            x_patch.shape[3] * self.patch_size,
        )
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1)
        self.to_hidden_dw = nn.Conv2d(
            dim * 6,
            dim * 6,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 6,
        )

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.norm = LayerNorm(dim * 2)
        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
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


class TransformerDownSampleBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_blocks,
        downsample=True,
        ffn_expansion_factor=2.66,
    ):
        super(TransformerDownSampleBlock, self).__init__()
        self.transformers = nn.Sequential(
            *list(
                chain(
                    *[
                        [LayerNorm(dim), DFFN(dim, ffn_expansion_factor)]
                        for n_blocks in range(n_blocks)
                    ]
                )
            )
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
                nn.Conv2d(dim, dim * 2, 3, stride=1, padding=1),
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        x = self.transformers(x)
        return self.downsample(x), x


class TransformerUpSampleBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_blocks,
        upsample=True,
        ffn_expansion_factor=2.66,
    ):
        super(TransformerUpSampleBlock, self).__init__()
        self.transformers = nn.Sequential(
            *list(
                chain(
                    *[
                        [
                            LayerNorm(dim),
                            FSAS(dim),
                            LayerNorm(dim),
                            DFFN(dim, ffn_expansion_factor),
                        ]
                        for n_blocks in range(n_blocks)
                    ]
                )
            )
        )
        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim, dim // 2, 3, stride=1, padding=1),
            )
        else:
            self.upsample = lambda x: x

    def forward(self, x):
        return self.upsample(self.transformers(x))


class FuseBlock(nn.Module):
    def __init__(self, n_feat):
        super(FuseBlock, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerDownSampleBlock(
            dim=n_feat * 2, n_blocks=1, downsample=False
        )
        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x, _ = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        return e + d


class FFTFormer(nn.Module):
    def __init__(
        self,
        channels=1,
        dim=48,
        num_blocks=[6, 6, 12],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
    ):
        super(FFTFormer, self).__init__()

        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=3, stride=1, padding=1)
        self.encoders = nn.ModuleList(
            [
                TransformerDownSampleBlock(
                    dim=dim * 2**ix,
                    n_blocks=n_blocks,
                    downsample=ix < len(num_blocks) - 1,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
                for ix, n_blocks in enumerate(num_blocks)
            ]
        )
        self.middle_layer = TransformerUpSampleBlock(
            dim=dim * 2 ** (len(num_blocks) - 1),
            n_blocks=num_blocks[-1],
            upsample=True,
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.decoders = nn.ModuleList(
            [
                TransformerUpSampleBlock(
                    dim=dim * 2**ix,
                    n_blocks=n_blocks,
                    upsample=ix != 0,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
                for ix, n_blocks in reversed(list(enumerate(num_blocks)))
            ][1:]
        )
        self.refinement = TransformerUpSampleBlock(
            dim=dim,
            n_blocks=num_refinement_blocks,
            upsample=False,
            ffn_expansion_factor=ffn_expansion_factor,
        )

        self.fuse_layers = nn.ModuleList(
            [FuseBlock(dim * 2**i) for i in range(len(num_blocks) - 1)]
        )
        self.output = nn.Conv2d(int(dim), channels, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        x = self.patch_embed(img.clone())
        encoder_outputs = []
        for encoder in self.encoders:
            x, y = encoder(x)
            encoder_outputs.append(y)
        decoded = self.middle_layer(encoder_outputs[-1])
        for encoded, decoder, fuse in zip(
            reversed(encoder_outputs[: len(self.decoders)]),
            self.decoders,
            reversed(self.fuse_layers),
        ):
            decoded = decoder(fuse(decoded, encoded))
        refined = self.refinement(decoded)
        return self.output(refined) + img
