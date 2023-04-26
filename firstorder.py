import os, time
import SimpleITK as sitk
import numpy as np
import radiomics
import ray


""" first-order mean """
def masked_mean_filter(image_arr, mask_arr, kernel_size):
    @ray.remote
    def process_unit(x_min, x_max, i, image_arr, mask_arr, kernel_size):
        slice_result = []
        for j in range(image_arr.shape[1]):
            for k in range(image_arr.shape[2]):
                if mask_arr[i, j, k] == 0:
                    slice_result.append(0)
                else:
                    y_min = max(0, j - kernel_size // 2)
                    y_max = min(image_arr.shape[1], j + kernel_size // 2 + 1)
                    z_min = max(0, k - kernel_size // 2)
                    z_max = min(image_arr.shape[2], k + kernel_size // 2 + 1)

                    kernel = image_arr[x_min:x_max, y_min:y_max, z_min:z_max]
                    kernel = kernel[mask_arr[x_min:x_max, y_min:y_max, z_min:z_max] > 0]

                    if len(kernel) == 0:
                        slice_result.append(0)
                    else:
                        slice_result.append(np.mean(kernel))

        return slice_result

    ray.init()
    image_arr_shared = ray.put(image_arr)
    mask_arr_shared = ray.put(mask_arr)
    kernel_size_shared = ray.put(kernel_size)

    futures = []
    for i in range(image_arr.shape[0]):
        # check if the slice is empty in the mask
        if np.sum(mask_arr[i, :, :]) > 0:
            x_min, x_max = max(0, i - kernel_size // 2), min(image_arr.shape[0], i + kernel_size // 2 + 1)
            futures.append(process_unit.remote(x_min, x_max, i, image_arr_shared, mask_arr_shared, kernel_size_shared))

    results = ray.get(futures)
    filtered_arr = np.zeros(image_arr.shape)
    for i in range(image_arr.shape[0]):
        if np.sum(mask_arr[i, :, :]) > 0:
            filtered_arr[i, :, :] = np.array(results.pop(0)).reshape(image_arr.shape[1], image_arr.shape[2])


    return filtered_arr


""" first-order Root Mean Square """
def masked_RMS_filter(image_arr, mask_arr, kernel_size):
    @ray.remote
    def process_unit(x_min, x_max, i, image_arr, mask_arr, kernel_size):
        slice_result = []
        for j in range(image_arr.shape[1]):
            for k in range(image_arr.shape[2]):
                if mask_arr[i, j, k] == 0:
                    slice_result.append(0)
                else:
                    y_min = max(0, j - kernel_size // 2)
                    y_max = min(image_arr.shape[1], j + kernel_size // 2 + 1)
                    z_min = max(0, k - kernel_size // 2)
                    z_max = min(image_arr.shape[2], k + kernel_size // 2 + 1)

                    kernel = image_arr[x_min:x_max, y_min:y_max, z_min:z_max]
                    kernel = kernel[mask_arr[x_min:x_max, y_min:y_max, z_min:z_max] > 0]

                    if len(kernel) == 0:
                        slice_result.append(0)
                    else:
                        slice_result.append(np.sqrt(np.mean(kernel ** 2)))

        return slice_result

    ray.init()
    image_arr_shared = ray.put(image_arr)
    mask_arr_shared = ray.put(mask_arr)
    kernel_size_shared = ray.put(kernel_size)

    futures = []
    for i in range(image_arr.shape[0]):
        # check if the slice is empty in the mask
        if np.sum(mask_arr[i, :, :]) > 0:
            x_min, x_max = max(0, i - kernel_size // 2), min(image_arr.shape[0], i + kernel_size // 2 + 1)
            futures.append(process_unit.remote(x_min, x_max, i, image_arr_shared, mask_arr_shared, kernel_size_shared))

    results = ray.get(futures)
    filtered_arr = np.zeros(image_arr.shape)
    for i in range(image_arr.shape[0]):
        if np.sum(mask_arr[i, :, :]) > 0:
            filtered_arr[i, :, :] = np.array(results.pop(0)).reshape(image_arr.shape[1], image_arr.shape[2])

    return filtered_arr


""" first-order Uniformity """
def masked_Uniformity_filter(image_arr, mask_arr, kernel_size):

    @ray.remote
    def process_unit(x_min, x_max, i, image_arr, mask_arr, kernel_size):
        slice_result = []
        for j in range(image_arr.shape[1]):
            for k in range(image_arr.shape[2]):
                if mask_arr[i, j, k] == 0:
                    slice_result.append(0)
                else:
                    y_min = max(0, j - kernel_size // 2)
                    y_max = min(image_arr.shape[1], j + kernel_size // 2 + 1)
                    z_min = max(0, k - kernel_size // 2)
                    z_max = min(image_arr.shape[2], k + kernel_size // 2 + 1)

                    kernel = image_arr[x_min:x_max, y_min:y_max, z_min:z_max]
                    kernel = kernel[mask_arr[x_min:x_max, y_min:y_max, z_min:z_max] > 0]

                    if len(kernel) == 0:
                        slice_result.append(0)
                    else:
                        slice_result.append(np.sum(kernel ** 2))

        return slice_result

    ray.init()
    image_arr_shared = ray.put(image_arr)
    mask_arr_shared = ray.put(mask_arr)
    kernel_size_shared = ray.put(kernel_size)

    futures = []
    for i in range(image_arr.shape[0]):
        # check if the slice is empty in the mask
        if np.sum(mask_arr[i, :, :]) > 0:
            x_min, x_max = max(0, i - kernel_size // 2), min(image_arr.shape[0], i + kernel_size // 2 + 1)
            futures.append(process_unit.remote(x_min, x_max, i, image_arr_shared, mask_arr_shared, kernel_size_shared))

    results = ray.get(futures)
    filtered_arr = np.zeros(image_arr.shape)
    for i in range(image_arr.shape[0]):
        if np.sum(mask_arr[i, :, :]) > 0:
            filtered_arr[i, :, :] = np.array(results.pop(0)).reshape(image_arr.shape[1], image_arr.shape[2])


    return filtered_arr