from typing import Tuple

import torch


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(EncoderBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )
        self.pooling = torch.nn.MaxPool2d((2, 2), padding=0)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(images)
        pooling_out = self.pooling(conv_out)
        return pooling_out, conv_out


class MiddleBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(MiddleBlock, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, None]:
        conv_out = self.conv(images)
        return conv_out, None


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(DecoderBlock, self).__init__()

        self.t_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_size, out_size, kernel_size=2, stride=2, padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(2 * out_size, out_size, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_size),
        )

    def forward(
        self, x: torch.Tensor, skipped_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.t_conv(x)
        conv_out = self.conv(torch.cat((x, skipped_x), 1))
        return conv_out


class NoiseDetector(torch.nn.Module):
    def __init__(self, channels=1, first_output=64, depth=5) -> None:
        super(NoiseDetector, self).__init__()
        self.encoder = torch.nn.ModuleList(
            [EncoderBlock(channels, first_output)]
            + [
                EncoderBlock(first_output * (2 ** (d - 1)), first_output * (2**d))
                for d in range(1, depth - 1)
            ]
            + [
                MiddleBlock(
                    first_output * (2 ** (depth - 2)), first_output * (2 ** (depth - 1))
                )
            ]
        )

        self.decoder = torch.nn.ModuleList(
            [
                DecoderBlock(first_output * (2**d), first_output * (2 ** (d - 1)))
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
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.decoder):
            before_pool = encoder_outs[-(i + 2)]
            x = module(x, before_pool)

        return self.output(x)
