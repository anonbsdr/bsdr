import numpy as np
import cv2
import os


def _gaussian_kernel(sigma=1.0, kernel_size=None):
    '''
    Returns gaussian kernel if sigma > 0.0, otherwise dot kernel.
    '''
    if sigma <= 0.0:
        return np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]], dtype=np.float32)
    if kernel_size is None:
        kernel_size = int(3.0 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
        print('In data_reader.gaussian_kernel: Kernel size even; ' \
              'increased by 1.')
    if kernel_size < 3:
        kernel_size = 3
        print('In data_reader.gaussian_kernel: Kernel size less than 3;' \
              'set as 3.')
    tmp = np.arange((-kernel_size // 2) + 1.0, (kernel_size // 2) + 1.0)
    xx, yy = np.meshgrid(tmp, tmp)
    kernel = np.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2)))
    kernel_sum = np.sum(kernel)
    assert (kernel_sum > 1e-3)
    return kernel / kernel_sum

def _create_heatmap(image_shape, heatmap_shape,
                   annotation_points, kernel):
    """
    Creates density map.
    annotation_points : ndarray Nx2,
                        annotation_points[:, 0] -> x coordinate
                        annotation_points[:, 1] -> y coordinate
    """
    assert (kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2
            and kernel.shape[0] > 1)
    indices = (annotation_points[:, 0] < image_shape[1]) & \
              (annotation_points[:, 0] >= 0) & \
              (annotation_points[:, 1] < image_shape[0]) & \
              (annotation_points[:, 1] >= 0)
    annot_error_count = len(annotation_points)
    annotation_points = annotation_points[indices, :]

    hmap_height, hmap_width = heatmap_shape
    annotation_points[:, 0] *= (1. * heatmap_shape[1] / image_shape[1])
    annotation_points[:, 1] *= (1. * heatmap_shape[0] / image_shape[0])
    annotation_points = annotation_points.astype(np.int32)
    annot_error_count -= np.sum(indices)
    if annot_error_count:
        print('In data_reader.create_heatmap: Error in annotations; ' \
              '%d point(s) skipped.' % annot_error_count)
    indices = (annotation_points[:, 0] >= heatmap_shape[1]) & \
              (annotation_points[:, 0] < 0) & \
              (annotation_points[:, 1] >= heatmap_shape[0]) & \
              (annotation_points[:, 1] < 0)
    assert(np.sum(indices) == 0)

    prediction_map = np.zeros(heatmap_shape, dtype = np.float32)
    kernel_half_size = kernel.shape[0] // 2
    kernel_copy = np.empty_like(kernel)

    for x, y in annotation_points:
        y_start = y - kernel_half_size
        y_end = y_start + kernel.shape[0]
        x_start = x - kernel_half_size
        x_end = x_start + kernel.shape[1]
        kernel_copy[:] = kernel[:]
        kernel_tmp = kernel_copy
        if y_start < 0:
            i = -y_start
            kernel_tmp[i: 2 * i, :] += kernel_tmp[i - 1:: -1, :]
            kernel_tmp = kernel_tmp[i:, :]
            y_start = 0
        if x_start < 0:
            i = -x_start
            kernel_tmp[:, i: 2 * i] += kernel_tmp[:, i - 1:: -1]
            kernel_tmp = kernel_tmp[:, i:]
            x_start = 0
        if y_end > hmap_height:
            i = (hmap_height - y - 1) - kernel_half_size
            kernel_tmp[2 * i: i, :] += kernel_tmp[-1: i - 1: -1, :]
            kernel_tmp = kernel_tmp[: i, :]
            y_end = hmap_height
        if x_end > hmap_width:
            i = (hmap_width - x - 1) - kernel_half_size
            kernel_tmp[:, 2 * i: i] += kernel_tmp[:, -1: i - 1: -1]
            kernel_tmp = kernel_tmp[:, : i]
            x_end = hmap_width
        prediction_map[y_start: y_end, x_start: x_end] += kernel_tmp
    return prediction_map
