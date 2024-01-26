# SPDX-FileCopyrightText: Copyright (c) 2019 mateuszbuda
#
# SPDX-License-Identifier: MIT
"""
These util functions are from original implementation
see: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/utils.py

A faster implementation of resize_sample (fast_resize_sample) can be used to
speed up testing.

---
https://raw.githubusercontent.com/mateuszbuda/brain-segmentation-pytorch/master/LICENSE
"""

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask


def fast_resize_sample(x, size=256):
    """
    Slight accuracy degradation, but great speed up.
    used the cv2.INTER_NEAREST interpolation for the mask since order=0
    corresponds to nearest-neighbor interpolation.
    For the volume, cv2.INTER_LINEAR as it corresponds to the
    second-order interpolation in the original function.
    """
    import cv2

    volume, mask = x
    v_shape = volume.shape

    # Resize the mask
    out_shape = (size, size)
    mask_resized = np.zeros((v_shape[0], size, size))
    for i in range(v_shape[0]):
        mask_resized[i] = cv2.resize(mask[i], out_shape, interpolation=cv2.INTER_NEAREST)

    # Resize the volume
    out_shape = out_shape + (v_shape[3],)
    volume_resized = np.zeros((v_shape[0], size, size, v_shape[3]))
    for i in range(v_shape[0]):
        for j in range(v_shape[3]):
            volume_resized[i, :, :, j] = cv2.resize(volume[i, :, :, j], out_shape[:2], interpolation=cv2.INTER_LINEAR)

    return volume_resized, mask_resized


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume
