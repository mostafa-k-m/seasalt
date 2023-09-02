import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from cpython cimport dict

cpdef dict calculate_distance_lookups(int size, int exp=2):
    cdef cnp.ndarray[long, ndim=2] all_kernel_positions = np.transpose(
        np.where(
            np.ones((size, size)) > 0
        )
    )
    cdef int center_ix = (size - 1) // 2
    cdef cnp.ndarray[double, ndim=1] distance_weights = (
        1
        / (
            1
            + (all_kernel_positions[:, 0] - center_ix) ** 2
            + (all_kernel_positions[:, 1] - center_ix) ** 2
        )
        ** exp
    )
    
    cdef dict lookup_dict = {}
    cdef float* weights_ptr = <float*>malloc(size * size * sizeof(float))
    cdef int ix
    
    for ix, pos in enumerate(all_kernel_positions):
        weights_ptr[ix] = distance_weights[ix]
    
    for ix in range(size):
        lookup_dict[ix] = {}
        for jx in range(size):
            lookup_dict[ix][jx] = weights_ptr[ix * size + jx]
    
    free(weights_ptr)
    
    return lookup_dict

cpdef float weighted_mean(
        cnp.ndarray[cnp.uint8_t, ndim=2] kernel,
        dict distance_lookup,
        int threshold=1
    ):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] selector = kernel + 1 > threshold
    
    cdef cnp.ndarray[long, ndim=2] ixs = np.transpose(np.where(kernel + 1 > threshold))
    cdef int num_ixs = ixs.shape[0]
    if num_ixs == 0:
        return np.median(kernel)

    cdef cnp.ndarray[double, ndim=1] distance_weights = np.empty(
        num_ixs, dtype=np.float64
    )
    for i in range(num_ixs):
        distance_weights[i] = distance_lookup[ixs[i, 0]][ixs[i, 1]]
    
    cdef float weighted_sum = np.sum(distance_weights * kernel[kernel + 1 > threshold])
    cdef float weights_sum = np.sum(distance_weights)
    
    return (
        weighted_sum / weights_sum if weights_sum > 0 else 0.0
    )

cpdef int get_dynamic_threshold(
        cnp.ndarray[cnp.uint8_t, ndim=2] arr,
        int size
    ):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] arr_copy = np.copy(arr).astype(np.uint8) + 1
    cdef tuple[int, cnp.ndarray[int, ndim=1]] result = np.histogram(
        arr, bins=range(0, 256, 5)
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] inflection_points = np.diff(
        result[0].tolist() + [0, 0]
    ) < 0
    if np.sum(result[0] > 0) >= 22:
        return (
            result[1][inflection_points][0] + 1
            if np.any(inflection_points)
            else result[1][0] + 1
        )
    return 2


cpdef cnp.ndarray[cnp.uint8_t, ndim=2] fixed_window_outlier_filter(
        cnp.ndarray[cnp.uint8_t, ndim=2] arr,
        int size=3,
        int exp=2
    ):
    assert size % 2 == 1, "Kernel Size Must be an Odd Number"
    cdef int kernel_center = (size - 1) // 2
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] padded_arr = np.pad(
        arr, 
        ((kernel_center, kernel_center), (kernel_center, kernel_center)),
        "edge"
    )
    cdef int threshold = np.clip(get_dynamic_threshold(arr, size), 2, 155)
    cdef dict distance_lookup = calculate_distance_lookups(size, exp)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] result = arr.copy()

    for index in np.argwhere(arr + 1 < threshold):
        result[index[0], index[1]] = weighted_mean(padded_arr[
            index[0] - size + 2 * kernel_center + 1 : index[0] + size,
            index[1] - size + 2 * kernel_center + 1 : index[1] + size,
        ], distance_lookup=distance_lookup, threshold=threshold)

    return result

