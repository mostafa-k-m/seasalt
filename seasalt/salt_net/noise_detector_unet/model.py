from typing import Tuple

import torch


class SqueezeExcitation(torch.nn.Module):
    def __init__(self, filter_size, ratio):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(filter_size, filter_size // ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(filter_size // ratio, filter_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.linear(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_size, out_size, squeeze_excitation, dropout) -> None:
        super(ConvLayer, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(out_size),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_size, out_size, kernel_size=3, stride=1, padding="same"
            ),
            torch.nn.BatchNorm2d(out_size, momentum=0.1, eps=1e-5),
            torch.nn.ReLU(),
        )

        if squeeze_excitation:
            self.squeeze_excitation = SqueezeExcitation(out_size, 8)
        else:
            self.squeeze_excitation = None

        if dropout:
            self.dropout = torch.nn.Dropout2d(p=0.2)
        else:
            self.dropout = None

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(images)
        if self.squeeze_excitation:
            conv_out = self.squeeze_excitation(conv_out)
        if self.dropout:
            conv_out = self.dropout(conv_out)
        return conv_out


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, squeeze_excitation, dropout) -> None:
        super(EncoderBlock, self).__init__()

        self.conv = ConvLayer(in_size, out_size, squeeze_excitation, dropout)
        self.pooling = torch.nn.MaxPool2d((2, 2), padding=0)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(images)
        downsampled = self.pooling(conv_out)
        return downsampled, conv_out


class MiddleBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, squeeze_excitation, dropout) -> None:
        super(MiddleBlock, self).__init__()

        self.conv = ConvLayer(in_size, out_size, squeeze_excitation, dropout)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, None]:
        conv_out = self.conv(images)
        return conv_out, None


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, squeeze_excitation, dropout) -> None:
        super(DecoderBlock, self).__init__()

        self.t_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2, padding=0
            ),
            torch.nn.BatchNorm2d(out_size),
            torch.nn.ReLU(),
        )

        self.conv = ConvLayer(2 * out_size, out_size, squeeze_excitation, dropout)

    def pad_on_upscale(self, x_1, x_2):
        return torch.nn.functional.pad(
            x_1,
            (
                0,
                max(
                    max(x_2.shape[-1], x_1.shape[-1])
                    - min(x_2.shape[-1], x_1.shape[-1]),
                    0,
                ),
                0,
                max(
                    max(x_2.shape[-2], x_1.shape[-2])
                    - min(x_2.shape[-2], x_1.shape[-2]),
                    0,
                ),
            ),
        )

    def forward(
        self, x: torch.Tensor, skipped_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.t_conv(x)
        x = self.pad_on_upscale(x, skipped_x)
        skipped_x = self.pad_on_upscale(skipped_x, x)
        conv_out = self.conv(torch.cat((x, skipped_x), 1))
        return conv_out


class NoiseDetectorUNet(torch.nn.Module):
    def __init__(
        self,
        channels=1,
        first_output=64,
        depth=5,
        max_exp=5,
        squeeze_excitation=False,
        dropout=False,
    ) -> None:
        super(NoiseDetectorUNet, self).__init__()
        self.encoder = torch.nn.ModuleList(
            [EncoderBlock(channels, first_output, squeeze_excitation, dropout)]
            + [
                EncoderBlock(
                    first_output * (2 ** (min(d - 1, max_exp))),
                    first_output * (2 ** min(d, max_exp)),
                    squeeze_excitation,
                    dropout,
                )
                for d in range(1, depth - 1)
            ]
            + [
                MiddleBlock(
                    first_output * (2 ** (min((depth - 2), max_exp))),
                    first_output * (2 ** min((depth - 1), max_exp)),
                    squeeze_excitation,
                    dropout,
                )
            ]
        )

        self.decoder = torch.nn.ModuleList(
            [
                DecoderBlock(
                    first_output * (2 ** min(d, max_exp)),
                    first_output * (2 ** (min(d - 1, max_exp))),
                    squeeze_excitation,
                    dropout,
                )
                for d in range(1, depth)
            ][::-1]
        )

        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                first_output, channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder_outs = []
        x = images

        for module in self.encoder:
            x, downsampled = module(x)
            encoder_outs.append(downsampled)

        for i, module in enumerate(self.decoder):
            upsampled = encoder_outs[-(i + 2)]
            x = module(x, upsampled)

        return self.output(x)
