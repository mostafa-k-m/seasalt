from typing import Dict

import numpy as np
from numpy.typing import NDArray


def image_histogram_equalization(arr):
    histogram, bins = np.histogram(
        np.ma.masked_where(
            arr.flatten(), (arr.flatten() == 0) | (arr.flatten() >= 255)
        ),
        256,
        density=True,
    )
    hist_cumsum = histogram.cumsum()
    hist_cumsum = np.ceil(255 * hist_cumsum / hist_cumsum[-1])
    eq_im = np.interp(arr.flatten(), bins[:-1], hist_cumsum).astype(np.uint8)
    return eq_im.reshape(arr.shape)


def get_dynamic_threshold(arr: NDArray[np.uint8], size: int) -> int:
    arr = (
        image_histogram_equalization(np.copy(arr)).astype(np.uint8) + 1  # type: ignore
    )
    counts, values = np.histogram(arr, bins=range(0, 256, 5))
    inflection_points = np.diff(counts.tolist() + [0, 0]) < 0
    return (
        values[inflection_points][0] + 2 if np.any(inflection_points) else values[0] + 2
    )


def weighted_mean(
    kernel: NDArray[np.uint8], distance_lookup, threshold: int = 1
) -> float:
    selector = kernel + 1 >= threshold
    ixs = np.transpose(np.where(selector))
    distance_weights = np.array([distance_lookup[ix[0]][ix[1]] for ix in ixs])
    if distance_weights.shape == (0,):
        return np.median(kernel)  # type: ignore
    return (
        np.sum(distance_weights * kernel[selector])  # type: ignore
        / np.sum(distance_weights)
        if (np.sum(distance_weights)) > 0  # type: ignore
        else 0
    )


def calculate_distance_lookups(
    size: int,
    exp: int = 2,
) -> Dict[int, Dict[int, float]]:
    all_kernel_positions = np.transpose(np.where(np.ones((size, size)) > 0))
    center_ix = int((size - 1) / 2)
    distance_weights = (
        1
        / (
            1
            + (all_kernel_positions[:, 0] - center_ix) ** 2
            + (all_kernel_positions[:, 1] - center_ix) ** 2
        )
        ** exp
    )
    lookup_dict = {i: {} for i in range(size)}
    for ix, pos in enumerate(all_kernel_positions):
        lookup_dict[pos[0]][pos[1]] = distance_weights[ix]
    return lookup_dict


def fixed_window_outlier_filter(
    arr: NDArray[np.uint8],
    size: int = 3,
    exp: int = 2,
) -> NDArray[np.uint8]:
    assert size % 2 == 1, "Kernel Size Must be an Odd Number"
    kernel_center = int((size - 1) / 2)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    padded_arr = np.pad(arr, (int((size - 1) / 2), int((size - 1) / 2)), "edge")
    threshold = get_dynamic_threshold(arr, size)
    distance_lookup = calculate_distance_lookups(size, exp)
    for index in np.argwhere(arr + 1 < threshold):
        arr[*index] = weighted_mean(
            padded_arr[
                index[0] - size + 2 * kernel_center + 1 : index[0] + size,
                index[1] - size + 2 * kernel_center + 1 : index[1] + size,
            ],
            threshold=threshold,
            distance_lookup=distance_lookup,
        )
    return arr
