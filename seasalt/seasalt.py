import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import medfilt

warnings.filterwarnings("ignore")


def apply_salt_pepper(arr: NDArray[np.uint8], ratio: float = 0.1) -> NDArray[np.uint8]:
    flat = arr.flatten()
    tot_number_of_pixels = len(flat)
    number_of_pixels_to_be_replaced = round(tot_number_of_pixels * ratio)
    pixels_to_be_replaced = np.random.choice(
        range(tot_number_of_pixels),
        size=(number_of_pixels_to_be_replaced,),
        replace=False,
    )
    flat[pixels_to_be_replaced[: round(number_of_pixels_to_be_replaced / 2)]] = 0.0
    flat[pixels_to_be_replaced[-round(number_of_pixels_to_be_replaced / 2) :]] = 255.0
    return np.uint8(flat.reshape(arr.shape))  # type: ignore


def get_kernel_slices(
    original_arr_indices: NDArray[np.uint8], size: int = 3, pad: bool = True
) -> tuple[slice, slice]:
    kernel_center = int((size - 1) / 2)
    x, y = original_arr_indices + (kernel_center if pad else 0)
    return slice(x - size + kernel_center + 1, x + size - kernel_center), slice(
        y - size + kernel_center + 1, y + size - kernel_center
    )


def filter_below_threshold(
    kernel: NDArray[np.uint8], threshold: int, size: int
) -> NDArray[np.uint8]:
    kernel_center = int((size - 1) / 2)
    mean = (
        (np.sum(kernel.astype(np.float64)) - kernel[kernel_center, kernel_center])
        / (size**2)
    ).astype(np.uint8)
    if np.abs(float(kernel[kernel_center, kernel_center] - mean)) > threshold:
        kernel[kernel_center, kernel_center] = mean
    return kernel.astype(np.uint8)


def get_dynamic_threshold(
    size: int, padded_arr: NDArray[np.uint8], indices_of_salt_pepper: NDArray[np.int64]
) -> int:
    all_kernels = list(
        filter(
            lambda x: x.shape == (size, size),
            [
                padded_arr[get_kernel_slices(salt_and_pepper_pixel, size)]
                for salt_and_pepper_pixel in indices_of_salt_pepper
            ],
        )
    )
    kernel_center = int((size - 1) / 2)
    kernel_means_array = list(
        map(
            partial(calc_mean, kernel_center),
            all_kernels,
        )
    )
    counts, values = np.histogram(
        kernel_means_array, bins=range(0, 256, 5)  # type: ignore
    )
    inflection_points = np.diff(counts.tolist() + [0, 0]) > 0
    return (
        values[inflection_points][0] + 5 if np.any(inflection_points) else values[0] + 5
    )


def fixed_window_outlier_filter(
    arr: NDArray[np.uint8], size: int = 3, mask: Optional[NDArray[np.bool_]] = None
) -> NDArray[np.uint8]:
    assert size % 2 == 1, "Kernel Size Must be an Odd Number"
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    padded_arr = np.pad(arr, (int((size - 1) / 2), int((size - 1) / 2)), "edge")
    if mask is not None:
        padded_mask = np.pad(mask, (int((size - 1) / 2), int((size - 1) / 2)), "edge")
        padded_arr = np.ma.masked_where(padded_mask, padded_arr)
    indices_of_salt_pepper = np.argwhere((arr == 0) | (arr == 255))
    threshold = get_dynamic_threshold(size, padded_arr, indices_of_salt_pepper)
    for index in indices_of_salt_pepper:
        slices = get_kernel_slices(index, size)
        padded_arr[slices] = filter_below_threshold(
            padded_arr[slices], threshold, size=size
        )
    return padded_arr[
        int((size - 1) / 2) : -int((size - 1) / 2),
        int((size - 1) / 2) : -int((size - 1) / 2),
    ]


def calc_mean(kernel_center: int, kernel: NDArray[np.uint8]) -> int:
    _mean = (
        np.mean(kernel.astype(np.float64)[(kernel > 0) & (kernel < 255)])
        - kernel[kernel_center, kernel_center]
    ).astype(np.uint8)
    return _mean if isinstance(_mean, np.uint8) else 0  # type: ignore


def coerce_to_array(im: NDArray[np.uint8] | Image.Image) -> NDArray[np.uint8]:
    return im if isinstance(im, np.ndarray) else np.array(im)


def coerce_to_PIL(im: NDArray[np.uint8] | Image.Image) -> Image.Image:
    return Image.fromarray(im) if isinstance(im, np.ndarray) else im  # type: ignore


def signal_to_noise_ratio(
    original: NDArray[np.uint8] | Image.Image, transformed: NDArray[np.uint8]
) -> float:
    # sourcery skip: assign-if-exp, reintroduce-else
    original = coerce_to_array(original)
    transformed = coerce_to_array(transformed)
    mse = np.mean((original.astype("int64") - transformed.astype("int64")) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10((255**2) / mse)


def plot_denoise(
    im: Image.Image,
    sp_ratio: float = 0.05,
    save_path: Optional[str] = None,
    size: int = 3,
) -> None:
    im_gs = im.convert("L")
    arr = np.array(im_gs)
    seasoned_arr = apply_salt_pepper(arr, ratio=sp_ratio)
    median_filter_corrected_image = medfilt(np.copy(seasoned_arr), kernel_size=size)
    corrected_image = fixed_window_outlier_filter(
        np.copy(seasoned_arr), size=size  # type: ignore
    )
    corrected_image_plus_anisotropic = anisotropic_diffusion(corrected_image, kappa=10)
    corrected_image_edges_method = pipe(np.copy(seasoned_arr), size=size)
    corrected_image_edges_method_plus_anisotropic = anisotropic_diffusion(
        pipe(np.copy(seasoned_arr), size=size), kappa=16
    )
    corrected_adaptive_window = adaptive_kernel_size(
        np.copy(seasoned_arr), correction_function=np.median
    )
    corrected_adaptive_window_reisz = adaptive_kernel_size(
        np.copy(seasoned_arr),
        max_size=size,
        correction_function=modified_riesz_mean,
    )
    corrected_adaptive_window_reisz_edge = pipe(
        np.copy(seasoned_arr), size=size, func=adaptive_kernel_size
    )

    noise_snr = signal_to_noise_ratio(im_gs, np.copy(seasoned_arr))
    aof_snr = signal_to_noise_ratio(im_gs, corrected_image)
    med_snr = signal_to_noise_ratio(im_gs, median_filter_corrected_image)
    aof_an_snr = signal_to_noise_ratio(im_gs, corrected_image_plus_anisotropic)
    aof_e_snr = signal_to_noise_ratio(im_gs, corrected_image_edges_method)
    aof_e_an_snr = signal_to_noise_ratio(
        im_gs, corrected_image_edges_method_plus_anisotropic
    )
    aof_adap_snr = signal_to_noise_ratio(im_gs, corrected_adaptive_window)
    aof_adap_riesz_snr = signal_to_noise_ratio(im_gs, corrected_adaptive_window_reisz)
    aof_adap_riesz_e_snr = signal_to_noise_ratio(
        im_gs, corrected_adaptive_window_reisz_edge
    )

    fig = plt.figure(
        figsize=(
            im_gs.size[0] * 1 / 40,
            im_gs.size[1] * 1 / 20,
        ),
        dpi=80,
    )
    gs = fig.add_gridspec(5, 2, hspace=0.12, wspace=0.08)
    axes = gs.subplots(sharex="col", sharey="row")
    list(map(lambda ax: ax.set_axis_off(), axes.flatten()))
    list(  # type: ignore
        map(
            lambda input: input[0].set_title(input[1]),
            zip(
                axes.flatten(),
                [
                    "Original",
                    f"Noisy Image {round(noise_snr, 2)} dB",
                    f"Median Filter {round(med_snr, 2)} dB",
                    f"DTOF {round(aof_snr, 2)} dB",
                    "ANIS.DTOF " f"{round(aof_an_snr, 2)} dB",
                    f"IRDTOF {round(aof_e_snr, 2)} dB",
                    "ANIS.IRDTOF " f"{round(aof_e_an_snr, 2)} dB",
                    "AWSDTOF" f"{round(aof_adap_snr, 2)} dB",
                    "AWSDTOF Riesz Mean\n" f"{round(aof_adap_riesz_snr, 2)} dB",
                    "AWSDTOF Riesz Mean Edge Method\n"
                    f"{round(aof_adap_riesz_e_snr, 2)} dB",
                ],
            ),
        )
    )
    axes[0][0].imshow(im_gs, cmap="gray", vmin=0, vmax=255)
    axes[0][1].imshow(seasoned_arr, cmap="gray", vmin=0, vmax=255)
    axes[1][0].imshow(median_filter_corrected_image, cmap="gray", vmin=0, vmax=255)
    axes[1][1].imshow(corrected_image, cmap="gray", vmin=0, vmax=255)
    axes[2][0].imshow(corrected_image_plus_anisotropic, cmap="gray", vmin=0, vmax=255)
    axes[2][1].imshow(corrected_image_edges_method, cmap="gray", vmin=0, vmax=255)
    axes[3][0].imshow(
        corrected_image_edges_method_plus_anisotropic, cmap="gray", vmin=0, vmax=255
    )
    axes[3][1].imshow(corrected_adaptive_window, cmap="gray", vmin=0, vmax=255)
    axes[4][0].imshow(corrected_adaptive_window_reisz, cmap="gray", vmin=0, vmax=255)
    axes[4][1].imshow(
        corrected_adaptive_window_reisz_edge, cmap="gray", vmin=0, vmax=255
    )
    if save_path:
        plt.axis("off")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close()


def anisotropic_diffusion(
    img: NDArray[np.uint8] | Image.Image,
    num_iterations: int = 10,
    delta_t: float = 0.1,
    kappa: int = 50,
) -> NDArray[np.uint8]:
    diffused_image = np.array(img).astype(np.float32)

    for _ in range(num_iterations):
        d_x = np.gradient(diffused_image, axis=1)
        d_y = np.gradient(diffused_image, axis=0)

        c = 1 / (1 + (np.sqrt(d_x**2 + d_y**2) / kappa) ** 2)

        d_xx = np.gradient(c * d_x, axis=1)
        d_yy = np.gradient(c * d_y, axis=0)

        diffused_image += delta_t * (d_xx + d_yy)

    return diffused_image.astype(np.uint8)


def get_c(img: NDArray[np.uint8] | Image.Image, kappa: int = 50) -> float:
    diffused_image = np.array(img).astype(np.float32)
    d_x = np.gradient(diffused_image, axis=1)
    d_y = np.gradient(diffused_image, axis=0)
    return 1 / (1 + (np.sqrt(d_x**2 + d_y**2) / kappa) ** 2)


def get_edges(img: NDArray[np.uint8] | Image.Image) -> NDArray[np.uint8]:
    c = get_c(img)
    c = cv2.GaussianBlur(c, (3, 3), 0)
    c[c > 0.9] = 0
    c[c != 0] = 1
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        c.astype(np.uint8), connectivity=8
    )
    edges_mask = np.zeros_like(c, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 100:
            edges_mask[labels == label] = 255
    return edges_mask


def skeletonize_custom(img: NDArray[np.uint8]) -> NDArray[np.bool_]:
    skel = np.zeros(img.shape, np.uint8)
    _, image_edit = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image_edit) > 0:
        image_edit = cv2.morphologyEx(image_edit, cv2.MORPH_CLOSE, element)
        eroded = cv2.erode(image_edit, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image_edit, temp)
        skel = cv2.bitwise_or(skel, temp)
        image_edit = eroded.copy()
    return cv2.GaussianBlur(skel.astype(np.uint8), (3, 3), 0).astype(bool)


def pipe(
    img: NDArray[np.uint8] | Image.Image,
    size: int,
    func: Callable = fixed_window_outlier_filter,
) -> NDArray[np.uint8]:
    corrected_img = func(img, size)
    edges_mask = get_edges(corrected_img)
    skeletonized_mask = skeletonize_custom(edges_mask)
    base = func(corrected_img, size, mask=skeletonized_mask)
    edges = func(corrected_img, size, mask=~skeletonized_mask).astype(np.uint8)
    return np.ma.filled(np.where(base.mask, edges, base))


def modified_riesz_mean(kernel: NDArray[np.uint8]) -> float:
    size = kernel.shape[0]
    center_ix = int((size - 1) / 2)
    ixs = np.transpose(np.where((kernel != 0) & (kernel != 255)))
    pw = (
        1
        / (1 + ((center_ix + 1 - ixs[:, 0]) ** 2 + (center_ix + 1 - ixs[:, 1]) ** 2))
        ** 2
    )

    numerator = np.sum(pw * kernel[(kernel != 0) & (kernel != 255)])
    denominator = np.sum(pw)
    return numerator / denominator if denominator > 0 else 0  # type: ignore


def adaptive_kernel_size(
    arr: NDArray[np.uint8],
    max_size: int = 9,
    correction_function: Callable = modified_riesz_mean,
    mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.uint8]:
    padded_arr = np.pad(arr, ((max_size, max_size), (max_size, max_size)), "symmetric")
    if mask is not None:
        arr = np.ma.masked_where(mask, arr)
        padded_mask = np.pad(
            mask, ((max_size, max_size), (max_size, max_size)), "symmetric"
        )
        padded_arr = np.ma.masked_where(padded_mask, padded_arr)
    m, n = padded_arr.shape
    indices_of_salt_pepper = np.argwhere(~((padded_arr != 0) & (padded_arr != 255)))
    for i, j in indices_of_salt_pepper[
        (1 + max_size <= indices_of_salt_pepper[:, 0])
        & (indices_of_salt_pepper[:, 0] < m - max_size)
        & (1 + max_size <= indices_of_salt_pepper[:, 1])
        & (indices_of_salt_pepper[:, 1] < n - max_size)
    ]:  ###
        for k_size in range(1, max_size + 1):
            kernel = padded_arr[i - k_size : i + k_size, j - k_size : j + k_size]
            count_0 = np.argwhere((kernel == 0)).shape[0]
            count_255 = np.argwhere((kernel == 255)).shape[0]
            kernel_center = padded_arr[i, j]
            if (
                max(count_0, count_255) < 2 * k_size**2
                and kernel_center
                in [
                    0,
                    255,
                ]
            ) or (k_size == max_size):
                arr[i - max_size, j - max_size] = correction_function(kernel)
                break
    return arr
