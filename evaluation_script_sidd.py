import warnings
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from rich.progress import Progress
from skimage.io import imsave
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from seasalt.salt_net import (
    NoiseType,
    SaltNetOneStageHandler,
    noise_adder_numpy,
    plot_before_after_and_original,
)

warnings.filterwarnings("ignore")

path_to_images_1 = Path(
    "/Users/mostafakm/Downloads/SIDD_Small_sRGB_Only/Data"
).resolve()
image_dataset = glob(str(path_to_images_1 / "*"))

model = SaltNetOneStageHandler(denoiser_path="./models/pytorch_dfwb_data_128_6.h5")

root_path = Path("eval").resolve()
eval_exp_name = "low_gauss"


def apply_transformation(
    seasoned_image,
    arr,
    save_path: Optional[str] = None,
    eval_exp_name=eval_exp_name,
):
    corrected_img = model.predict(seasoned_image).astype(np.uint8)
    if save_path:
        plt.axis("off")
        full_save_path = root_path / eval_exp_name / save_path
        full_save_path.mkdir(parents=True, exist_ok=True)
        plot_before_after_and_original(
            arr, (seasoned_image).astype(np.uint8), corrected_img
        )
        plt.savefig(full_save_path / "transformation.png")
        plt.close()
        imsave(full_save_path / "corrected.png", corrected_img)
    return arr, corrected_img


def plot_exp_snr(df_eq, metric, dataset_save_path):
    df_aof_agg = df_eq.groupby("im_index").agg(
        {
            metric: [np.mean],
        }
    )
    df_aof_agg = df_aof_agg.reset_index()
    df_aof_agg.columns = ["im_index", "mean_metric"]
    df_melted = df_aof_agg.melt(
        id_vars="im_index", var_name="variable", value_name="value"
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
        x="im_index",
        y="value",
        hue="variable",
    )
    plt.xlabel("Noise Parameter")
    plt.ylabel(f"{metric}")
    plt.savefig(dataset_save_path / f"{eval_exp_name}_{metric}.png")
    plt.close()


def run_evaluation_loop(image_datasets):
    data_dict_psnr = dict(im_index=[], PSNR=[])
    data_dict_ssim = dict(im_index=[], SSIM=[])
    with Progress() as progress:
        task = progress.add_task("Running Evalution...", total=len(image_datasets))
        for image_paths in image_datasets:
            images = sorted(glob(image_paths + "/*"))
            img_name = images[-1].split("/")[-2]
            im_noisy = Image.open(images[-1]).convert("L")
            im_noisy = np.array(
                im_noisy.resize((im_noisy.size[0] // 4, im_noisy.size[1] // 4))
            )
            im = Image.open(images[-2]).convert("L")
            im = np.array(im.resize((im.size[0] // 4, im.size[1] // 4)))
            arr, corrected_img = apply_transformation(
                im_noisy,
                im,
                save_path=f"SIDD/{img_name}",
            )
            arr = arr[: corrected_img.shape[0], : corrected_img.shape[1]]
            data_dict_psnr["im_index"].append(img_name)
            data_dict_psnr["PSNR"].append(peak_signal_noise_ratio(arr, corrected_img))
            data_dict_ssim["im_index"].append(img_name)
            data_dict_ssim["SSIM"].append(structural_similarity(arr, corrected_img))
            progress.update(task, advance=1)
            sns.set_theme()
            dataset_save_path = root_path / eval_exp_name / f"SIDD/{img_name}"
        save_agg_results(data_dict_psnr, dataset_save_path, "PSNR")
        save_agg_results(data_dict_ssim, dataset_save_path, "SSIM")


def save_agg_results(data_dict_eq, dataset_save_path, metric):
    df_eq = pd.DataFrame(data_dict_eq)
    df_eq.to_pickle(dataset_save_path / f"df_{metric}.pkl")
    df_eq.to_excel(dataset_save_path / f"df_{metric}.xlsx", index=False)
    plot_exp_snr(df_eq, metric, dataset_save_path)


run_evaluation_loop(image_dataset)
