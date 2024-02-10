import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _flatten_grayscale_img(image):
    if len(image.shape) > 2:
        if image.shape[0] == 1:
            return np.reshape(image, (image.shape[1], image.shape[2]))
    return image


def _get_figure_dims(image, num_of_images):
    if len(image.shape) > 2:
        h = image.shape[1]
        w = image.shape[2]
    else:
        h = image.shape[0]
        w = image.shape[1]
    if h >= w:
        return (num_of_images * w / 100, h / 100)
    else:
        return (num_of_images * w / 100, h / 100)


def plot_single_image(img, m=255):
    img = _flatten_grayscale_img(img).astype(np.uint8)
    fig = plt.figure(
        figsize=_get_figure_dims(img, 1),
        dpi=100,
    )
    gs = fig.add_gridspec(1, 1, hspace=0, wspace=0)
    ax = gs.subplots(sharex="col", sharey="row")
    ax.imshow(img, cmap="gray", vmin=0, vmax=m)
    ax.set_axis_off()


def plot_before_and_after(before_img, after_img, m=255):
    before_img = _flatten_grayscale_img(before_img).astype(np.uint8)
    after_img = _flatten_grayscale_img(after_img).astype(np.uint8)
    before_img = before_img[: after_img.shape[0], : after_img.shape[1]]
    w, h = _get_figure_dims(after_img, 2)
    fig = plt.figure(
        figsize=(w, h),
        dpi=100,
    )
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[w * 100 / 2] * 2,
        hspace=0,
        wspace=0,
    )
    axes = gs.subplots(sharex="col", sharey="row")
    list(map(lambda ax: ax.set_axis_off(), axes.flatten()))
    list(  # type: ignore
        map(
            lambda input: input[0].set_title(input[1]),
            zip(
                axes.flatten(),
                [
                    "Before Image",
                    "After Image",
                ],
            ),
        )
    )
    axes[0].imshow(before_img, cmap="gray", vmin=0, vmax=m)
    axes[1].imshow(after_img, cmap="gray", vmin=0, vmax=m)


def plot_before_after_and_original(orginal, before_img, after_img, m=255):
    orginal = _flatten_grayscale_img(orginal).astype(np.uint8)
    before_img = _flatten_grayscale_img(before_img).astype(np.uint8)
    after_img = _flatten_grayscale_img(after_img).astype(np.uint8)
    orginal = orginal[: after_img.shape[0], : after_img.shape[1]]
    before_img = before_img[: after_img.shape[0], : after_img.shape[1]]
    w, h = _get_figure_dims(after_img, 3)
    fig = plt.figure(
        figsize=(w, h),
        dpi=100,
    )
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[w * 100 / 3] * 3,
        hspace=0,
        wspace=0,
    )
    axes = gs.subplots(sharex="col", sharey="row")
    list(map(lambda ax: ax.set_axis_off(), axes.flatten()))
    list(  # type: ignore
        map(
            lambda input: input[0].set_title(input[1]),
            zip(
                axes.flatten(),
                [
                    "Original Image",
                    "Before Image "
                    f"PSNR: {round(peak_signal_noise_ratio(orginal, before_img), 1)} dB"
                    f", SSIM: {round(structural_similarity(orginal, before_img), 4)}",
                    "After Image "
                    f"PSNR: {round(peak_signal_noise_ratio(orginal, after_img), 1)} dB"
                    f", SSIM: {round(structural_similarity(orginal, after_img), 4)}",
                ],
            ),
        )
    )
    axes[0].imshow(orginal, cmap="gray", vmin=0, vmax=m)
    axes[1].imshow(before_img, cmap="gray", vmin=0, vmax=m)
    axes[2].imshow(after_img, cmap="gray", vmin=0, vmax=m)
