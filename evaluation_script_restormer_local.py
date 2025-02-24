import warnings
from glob import glob
from math import floor
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rich.progress import Progress
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from seasalt.salt_net import (
    NoiseType,
    # SaltNetOneStageHandler,
    # noise_adder_numpy,
    noise_adder_numpy,
    plot_before_after_and_original,
)

warnings.filterwarnings("ignore")


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1
        )
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=False
        )
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
    ):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(
            1, channels[0], kernel_size=3, padding=1, bias=False
        )

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(num_ch, num_ah, expansion_factor)
                        for _ in range(num_tb)
                    ]
                )
                for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
            ]
        )
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList(
            [UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]]
        )
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(range(2, len(channels)))
            ]
        )
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(channels[2], num_heads[2], expansion_factor)
                        for _ in range(num_blocks[2])
                    ]
                )
            ]
        )
        self.decoders.append(
            nn.Sequential(
                *[
                    TransformerBlock(channels[1], num_heads[1], expansion_factor)
                    for _ in range(num_blocks[1])
                ]
            )
        )
        # the channel of last one is not change
        self.decoders.append(
            nn.Sequential(
                *[
                    TransformerBlock(channels[1], num_heads[0], expansion_factor)
                    for _ in range(num_blocks[0])
                ]
            )
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(channels[1], num_heads[0], expansion_factor)
                for _ in range(num_refinement)
            ]
        )
        self.output = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](
            self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1))
        )
        out_dec2 = self.decoders[1](
            self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1))
        )
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out


model = Restormer()
model.to("cpu")
model.load_state_dict(
    torch.load("./models/pytorch_restormer_gauss_only_blind_5_to_60_257.h5", "cpu")
)


def np_to_torch_even_shape(arr):
    x = torch.tensor(arr).unsqueeze(0).float()
    return x[:, :, : floor(x.shape[2] / 8) * 8, : floor(x.shape[3] / 8) * 8]


def predict_restormer(arr):
    tensor = np_to_torch_even_shape(arr)
    tensor = tensor.to("cpu") / 255
    tensor = model.forward(tensor)
    return tensor.squeeze().cpu().detach().numpy() * 255


path_to_images_1 = Path("data/Kodak24").resolve()
path_to_images_2 = Path("data/BSD68").resolve()
path_to_images_3 = Path("data/Set12").resolve()
path_to_images_4 = Path("data/Urban100/X2 Urban100/X2/HIGH X2 Urban").resolve()
image_datasets = [
    glob(str(path_to_images_1 / "*")),
    glob(str(path_to_images_2 / "*")),
    glob(str(path_to_images_3 / "*")),
    glob(str(path_to_images_4 / "*")),
]

# model = SaltNetOneStageHandler(denoiser_path="./models/pytorch_dfwb_data_128_6.h5")

root_path = Path("eval").resolve()
eval_exp_name = "restormer_all_noise_fully_trained"


def apply_transformation(
    im,
    noise_parameter,
    noise_type,
    save_path: Optional[str] = None,
    eval_exp_name=eval_exp_name,
):
    im_gs = im.convert("L")
    arr = np.array(im_gs)
    seasoned_image = noise_adder_numpy(arr, noise_parameter, noise_type)
    corrected_img = predict_restormer(seasoned_image).astype(np.uint8)
    if save_path:
        plt.axis("off")
        full_save_path = root_path / eval_exp_name / save_path
        full_save_path.mkdir(parents=True, exist_ok=True)
        plot_before_after_and_original(arr, seasoned_image, corrected_img)
        plt.savefig(full_save_path / "transformation.png")
        plt.close()
        imsave(full_save_path / "corrected.png", corrected_img)
    return arr, corrected_img


def plot_exp_snr(df_eq, metric, dataset_save_path):
    df_aof_agg = df_eq.groupby("noise_parameter").agg(
        {
            metric: [np.mean],
        }
    )
    df_aof_agg = df_aof_agg.reset_index()
    df_aof_agg.columns = ["noise_parameter", "mean_metric"]
    df_melted = df_aof_agg.melt(
        id_vars="noise_parameter", var_name="variable", value_name="value"
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_melted[
            df_melted.variable.isin(
                [
                    "mean_metric",
                ]
            )
        ],
        x="noise_parameter",
        y="value",
        hue="variable",
    )
    plt.xlabel("Noise Parameter")
    plt.ylabel(f"{metric}")
    plt.savefig(dataset_save_path / f"{eval_exp_name}_{metric}.png")
    plt.close()


def run_evaluation_loop(image_datasets, noise_type, noise_parameters):
    with Progress() as progress:
        for ix, image_paths in enumerate(image_datasets):
            task = progress.add_task(
                f"Running {noise_type.name} Evalution (Dataset {ix})...",
                total=len(image_paths) * len(noise_parameters),
            )
            data_dict_psnr = dict(im_index=[], noise_parameter=[], PSNR=[])
            data_dict_ssim = dict(im_index=[], noise_parameter=[], SSIM=[])
            for p in image_paths:
                img_name = p.split("/")[-1].split(".")[0]
                for noise_parameter in noise_parameters:
                    im = Image.open(p)
                    arr, corrected_img = apply_transformation(
                        im,
                        noise_parameter=noise_parameter / 255,
                        noise_type=noise_type,
                        save_path=f"{noise_type.name}/{p.split('data/')[1].split('/')[0]}/"
                        f"{p.split('/')[-1].split('.')[0]}"
                        f"/{noise_parameter}",
                    )
                    arr = arr[: corrected_img.shape[0], : corrected_img.shape[1]]
                    data_dict_psnr["im_index"].append(img_name)
                    data_dict_psnr["noise_parameter"].append(noise_parameter)
                    data_dict_psnr["PSNR"].append(
                        peak_signal_noise_ratio(arr, corrected_img)
                    )
                    data_dict_ssim["im_index"].append(img_name)
                    data_dict_ssim["noise_parameter"].append(noise_parameter)
                    data_dict_ssim["SSIM"].append(
                        structural_similarity(arr, corrected_img)
                    )
                    progress.update(task, advance=1)
            sns.set_theme()
            dataset_save_path = (
                root_path
                / eval_exp_name
                / noise_type.name
                / f"{p.split('data/')[1].split('/')[0]}"
            )
            save_agg_results(data_dict_psnr, dataset_save_path, "PSNR")
            save_agg_results(data_dict_ssim, dataset_save_path, "SSIM")


def save_agg_results(data_dict_eq, dataset_save_path, metric):
    df_eq = pd.DataFrame(data_dict_eq)
    df_eq.to_pickle(dataset_save_path / f"df_{metric}.pkl")
    df_eq.to_excel(dataset_save_path / f"df_{metric}.xlsx", index=False)
    plot_exp_snr(df_eq, metric, dataset_save_path)


run_evaluation_loop(
    image_datasets, NoiseType.GUASSIAN, noise_parameters=[15, 25, 50, 60]
)
run_evaluation_loop(
    image_datasets,
    NoiseType.SAP,
    noise_parameters=[150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
)
run_evaluation_loop(
    image_datasets,
    NoiseType.BERNOULLI,
    noise_parameters=[150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
)
run_evaluation_loop(
    image_datasets,
    NoiseType.POISSON,
    noise_parameters=[150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
)
