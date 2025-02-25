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

model = SaltNetOneStageHandler(
    denoiser_path="./models/pytorch_transformers_instead_of_cnn_64_36.h5",
    enable_seconv=True,
    enable_fft=False,
    enable_anisotropic=True,
    enable_unet_post_processing=True,
    transformer_depth=10,
)

root_path = Path("eval").resolve()
eval_exp_name = "transformers_instead_of_cnn_64"


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

# experiments = ["eval/lightweight_model_sigmoid_t"]
# xlsx_registery = {}
# for exp in experiments:
#     gl = glob(exp + "/**/**/*.xlsx")
#     exp_name = exp.split("/")[-1]
#     xlsx_registery[exp_name] = {}
#     for pth in gl:
#         noise_type = pth.split(exp + "/")[-1].split("/")[0]
#         dataset_name = pth.split(exp + "/" + noise_type + "/")[-1].split("/")[0]
#         df = pd.read_excel(pth)
#         if noise_type not in xlsx_registery[exp_name]:
#             xlsx_registery[exp_name][noise_type] = {}
#         if dataset_name not in xlsx_registery[exp_name][noise_type]:
#             xlsx_registery[exp_name][noise_type] = xlsx_registery[exp_name][
#                 noise_type
#             ] | {dataset_name: df}
#         else:
#             df = df.merge(xlsx_registery[exp_name][noise_type][dataset_name])
#             xlsx_registery[exp_name][noise_type][dataset_name] = df[
#                 ["im_index", "noise_parameter", "SSIM", "PSNR"]
#             ].copy()

# for exp in experiments:
#     exp = exp.split("eval/")[-1]
#     for noise_type in xlsx_registery[exp].keys():
#         df_list = []
#         for dataset in xlsx_registery[exp][noise_type]:
#             df = xlsx_registery[exp][noise_type][dataset]
#             df.insert(loc=0, column="dataset", value=dataset)
#             df_list.append(df)
#         df = pd.concat(df_list)
#         df.insert(loc=0, column="noise_type", value=noise_type)
#         xlsx_registery[exp][noise_type] = df
#     xlsx_registery[exp] = pd.concat(
#         [xlsx_registery[exp][noise_type] for noise_type in xlsx_registery[exp].keys()]
#     )
#     xlsx_registery[exp].to_excel(f"eval/{exp}.xlsx", index=False)
