import torch
import torch.nn.functional as F

from ..denoise_seconvnet import ConvBlock, OutputBlock, SeConvBlock
from ..noise_detector_unet import AutoEncoder, ConvLayer


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


class NormReshapeLayer(torch.nn.Module):
    def __init__(self, filters):
        super(NormReshapeLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(filters)

    def forward(self, x):
        b, c, h, w = x.shape
        normalized = self.norm(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
        return normalized.transpose(-2, -1).contiguous().reshape(b, c, h, w)


class MDTA(torch.nn.Module):
    def __init__(self, filters, attn_heads):
        super(MDTA, self).__init__()
        self.num_heads = attn_heads
        self.temperature = torch.nn.Parameter(torch.ones(attn_heads, 1, 1))
        self.qkv = torch.nn.Conv2d(filters, filters * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = torch.nn.Conv2d(
            filters * 3,
            filters * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=filters * 3,
            bias=False,
        )
        self.project_out = torch.nn.Conv2d(filters, filters, kernel_size=1, bias=False)

    def get_qkv(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        return k, q, v

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.get_qkv(x)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = attn @ v
        return self.project_out(out.reshape(b, -1, h, w))


class GDFN(torch.nn.Module):
    def __init__(self, filters, chnl_expansion_factor):
        super(GDFN, self).__init__()
        hidden_features = int(filters * chnl_expansion_factor)
        self.project_in = torch.nn.Conv2d(
            filters, hidden_features * 2, kernel_size=1, bias=False
        )
        self.conv = torch.nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=False,
        )
        self.project_out = torch.nn.Conv2d(
            hidden_features, filters, kernel_size=1, bias=False
        )

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        filters,
        attn_heads,
        chnl_expansion_factor,
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = NormReshapeLayer(filters)
        self.attn = MDTA(filters, attn_heads)
        self.norm2 = NormReshapeLayer(filters)
        self.ffn = GDFN(filters, chnl_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DenoiseNet(torch.nn.Module):

    def __init__(
        self,
        channels=1,
        auto_encoder_depth=5,
        seconv_depth=7,
        fft_depth=7,
        anisotropic_depth=5,
        transformer_depth=10,
        filters=48,
        chnl_expansion_factor=2.66,
        enable_seconv=True,
        enable_fft=False,
        enable_anisotropic=True,
        enable_unet_post_processing=True,
    ) -> None:
        super(DenoiseNet, self).__init__()
        self.enable_seconv = enable_seconv
        self.enable_fft = enable_fft
        self.enable_anisotropic = enable_anisotropic
        self.enable_unet_post_processing = enable_unet_post_processing

        n_outputs = 1

        if self.enable_seconv:
            n_outputs += 1
            self.seconv_blocks = torch.nn.ModuleList(
                [
                    SeConvBlock(kernel_size=7 + 2 * d, channels=channels)
                    for d in range(seconv_depth)
                ]
            )
            self.seconv_post_processing = OutputBlock(channels, channels)

        if self.enable_fft:
            n_outputs += 1
            self.fft_blocks = torch.nn.ModuleList(
                [FFTBlock(channels, filters) for _ in range(fft_depth)]
            )

        if self.enable_anisotropic:
            n_outputs += 1
            self.anisotropic_blocks = torch.nn.ModuleList(
                [
                    AnisotropicDiffusionBlock(channels=channels, kernel_size=3 + 2 * d)
                    for d in range(anisotropic_depth)
                ]
            )

        self.cnn_embeddings_layer = ConvBlock(n_outputs * channels, filters)

        if transformer_depth:
            self.transformer_layers = torch.nn.ModuleList(
                [
                    TransformerBlock(
                        filters=filters,
                        attn_heads=2 ** ((d + 1) // 3),
                        chnl_expansion_factor=chnl_expansion_factor,
                    )
                    for d in range(transformer_depth)
                ]
            )
        else:
            self.transformer_layers = None

        if self.enable_unet_post_processing:
            self.unet_post_processing = AutoEncoder(
                channels, filters, auto_encoder_depth, create_embeddings=False
            )
        else:
            self.output_layer = ConvLayer(
                in_size=filters,
                out_size=channels,
                squeeze_excitation=True,
                dropout=False,
            )

    def forward(self, noisy_images: torch.Tensor) -> torch.Tensor:
        outputs = [noisy_images.clone()]
        if self.enable_seconv:
            x_seconv = noisy_images
            for module in self.seconv_blocks:
                x_seconv = module(x_seconv)
            x_seconv = self.seconv_post_processing(outputs[0].clone(), x_seconv)
            outputs.append(x_seconv)
        if self.enable_fft:
            x_fft = noisy_images
            for module in self.fft_blocks:
                x_fft = module(x_fft)
            outputs.append(x_fft)
        if self.enable_anisotropic:
            x_anisotropic = noisy_images
            for module in self.anisotropic_blocks:
                x_anisotropic = module(x_anisotropic)
            outputs.append(x_anisotropic)
        embeddings = self.cnn_embeddings_layer(torch.cat(outputs, 1))
        if self.transformer_layers:
            for module in self.transformer_layers:
                embeddings = embeddings + module(embeddings)
        if self.enable_unet_post_processing:
            output = self.unet_post_processing(embeddings)
        else:
            output = self.output_layer(embeddings)
        return output
