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

path_to_images_1 = Path("TTI20_OTHER").resolve()
path_to_images_2 = Path("BSD68_OTHER").resolve()
image_datasets = [glob(str(path_to_images_1 / "*")), glob(str(path_to_images_2 / "*"))]

model = SaltNetOneStageHandler(denoiser_path="hybrid_non_gaussian2.pt")

root_path = Path("eval").resolve()
eval_exp_name = "hybrid_model"


def apply_transformation(
    im,
    noise_parameter,
    noise_type,
    save_path: Optional[str] = None,
    eval_exp_name=eval_exp_name,
):
    im_gs = im.convert("L")
    arr = np.array(im_gs)
    seasoned_image = noise_adder_numpy(arr, noise_parameter, noise_type) / 255
    corrected_img = model.predict(seasoned_image).astype(np.uint8)
    if save_path:
        plt.axis("off")
        full_save_path = root_path / eval_exp_name / save_path
        full_save_path.mkdir(parents=True, exist_ok=True)
        plot_before_after_and_original(
            arr, (seasoned_image * 255).astype(np.uint8), corrected_img
        )
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


def run_evaluation_loop(image_datasets, noise_type):
    data_dict_psnr = dict(im_index=[], noise_parameter=[], PSNR=[])
    data_dict_ssim = dict(im_index=[], noise_parameter=[], SSIM=[])
    with Progress() as progress:
        for ix, image_paths in enumerate(image_datasets):
            task = progress.add_task(
                f"Running {noise_type} Evalution (Dataset {ix})...",
                total=len(image_paths) * 10,
            )
            for p in image_paths:
                img_name = p.split("/")[-1].split(".")[0]
                for noise_parameter in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    im = Image.open(p)
                    arr, corrected_img = apply_transformation(
                        im,
                        noise_parameter=noise_parameter,
                        noise_type=noise_type,
                        save_path=f"{p.split('/')[-2].split('_')[0]}/"
                        f"{p.split('/')[-1].split('.')[0]}"
                        f"/{int(noise_parameter*100)}",
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
                root_path / eval_exp_name / f'{p.split("/")[-2].split("_")[0]}'
            )
            save_agg_results(data_dict_psnr, dataset_save_path, "PSNR")
            save_agg_results(data_dict_ssim, dataset_save_path, "SSIM")


def save_agg_results(data_dict_eq, dataset_save_path, metric):
    df_eq = pd.DataFrame(data_dict_eq)
    df_eq.to_pickle(dataset_save_path / f"df_{metric}.pkl")
    df_eq.to_excel(dataset_save_path / f"df_{metric}.xlsx", index=False)
    plot_exp_snr(df_eq, metric, dataset_save_path)


run_evaluation_loop(image_datasets, NoiseType.SAP)
run_evaluation_loop(image_datasets, NoiseType.BERNOULLI)
run_evaluation_loop(image_datasets, NoiseType.POISSON)
run_evaluation_loop(image_datasets, NoiseType.GUASSIAN)
