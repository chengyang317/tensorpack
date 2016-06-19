# -*- coding: UTF-8 -*-
# File: pad.py
# Author: philipcheng
# Time: 6/17/16 -> 10:03 PM
import numpy as np
from tensorpack.dataflow.imgaug.base import *

__all__ = ['Padding']


def _shape2d(shape):
    if type(shape) not in (tuple, list):
        return [shape] * 2
    ndim = len(shape)
    assert ndim in (1, 2)
    if ndim == 1:
        return list(shape) * 2
    else:
        return list(shape)


def _shape3d(shape, c_nums):
    if type(shape) not in (tuple, list):
        return [shape, shape, c_nums]
    ndim = len(shape)
    assert ndim in (1, 2, 3)
    if ndim == 1:
        return list(shape) * 2 + [c_nums]
    elif ndim == 2:
        return list(shape) + [c_nums]
    else:
        return list(shape)


class Padding(ImageAugmentor):
    """
    Pad a image
    """
    def __init__(self, target_shape, pad_value):
        """

        :param targe_shape:
        """
        self._init(locals())

    def _get_augment_params(self, img):
        origin_shape = img.shape
        target_shape = self.target_shape
        o_ndim = len(origin_shape)
        t_ndim = len(target_shape)
        assert o_ndim in (2, 3)
        assert o_ndim >= t_ndim
        if o_ndim != t_ndim:
            if o_ndim == 2:
                target_shape = _shape2d(target_shape)
            else:
                target_shape = _shape3d(target_shape, origin_shape[2])
        pad_nums = map(lambda x, y: max(0, x-y), origin_shape, target_shape)
        pad_width = zip([0] * len(origin_shape), pad_nums)
        return pad_width, self.pad_value

    def _augment(self, img, param):
        pad_width, pad_value = param
        return np.pad(img, pad_width=pad_width, mode='constant', constant_values=pad_value)
