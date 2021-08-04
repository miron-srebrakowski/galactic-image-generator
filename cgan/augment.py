import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.ndimage as ndi
from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom

from astropy.io import fits
from PIL import ImageEnhance, Image

augments = {}
augments['rotation_range'] = 90
augments['height_shift_range'] = 0.05
augments['width_shift_range'] = 0.05
augments['zoom_range'] = [0.9, 1.1]
augments['horizontal_flip'] = True
augments['vertical_flip'] = True


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest'):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def flip_axis(x, axis):  # TODO: remove?
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def aug(x, instruct = augments):
    # x is a single image, so it doesn't have image number at index 0
    # slightly tailored from Keras source code (Francois, Keras author)
    img_row_index = 0
    img_col_index = 1
    img_channel_index = 2

    # use composition of homographies to generate final transform that needs to be applied
    if instruct['rotation_range']:
        theta = np.pi / 180 * np.random.uniform(-instruct['rotation_range'], instruct['rotation_range'])
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if instruct['height_shift_range']:
        tx = np.random.uniform(-instruct['height_shift_range'], instruct['height_shift_range']) * x.shape[img_row_index]
    else:
        tx = 0

    if instruct['width_shift_range']:
        ty = np.random.uniform(-instruct['width_shift_range'], instruct['width_shift_range'] * x.shape[img_col_index])
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    
    shear_matrix = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])

    if instruct['zoom_range'][0] == 1 and instruct['zoom_range'][1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(instruct['zoom_range'][0], instruct['zoom_range'][1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index, fill_mode='wrap')
    
    if instruct['horizontal_flip']:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)

    if instruct['vertical_flip']:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)
    
    return x
