import torch


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


class Desnoiser(torch.nn.Module):
    def __init__(
        self, channels=1, seconv_depth=5, conv_depth=10, max_filters=128
    ) -> None:
        super(Desnoiser, self).__init__()
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
