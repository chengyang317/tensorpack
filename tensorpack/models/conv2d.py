#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
from ._common import *
from tensorpack.tfutils.filler import *

__all__ = ['Conv2D']

@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           w_init_config=None, b_init_config=None,
           nl=tf.nn.relu, split=1, use_bias=True):
    """
    2D convolution on 4D inputs.

    :param input: a tensor of shape NHWC
    :param kernel_shape: (h, w) or a int
    :param stride: (h, w) or a int. default to 1
    :param padding: 'valid' or 'same'. default to 'same'
    :param split: split channels as used in Alexnet. a int default to 1
    :param w_init_config:
    :param b_init_config: initializer for b. default to zero initializer.
    :param nl: nonlinearity. default to `relu`.
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a NHWC tensor
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride)

    if not w_init_config:
        w_init_config = FillerConfig(type='xavier', variance_norm='AVERAGE')
    if not b_init_config:
        b_init_config = FillerConfig(type='constant', value=0.0)
    w_initializer = get_initializer(filter_shape, w_init_config)
    b_initializer = get_initializer([out_channel], b_init_config)

    W = tf.get_variable('W', filter_shape, initializer=w_initializer)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_initializer)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding)
    else:
        inputs = tf.split(3, split, x)
        kernels = tf.split(3, split, W)
        outputs = [tf.nn.conv2d(i, k, stride, padding)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(3, outputs)
    return nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')


