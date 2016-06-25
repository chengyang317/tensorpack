#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
from tensorpack.models.utils import layer, shape2d, shape4d
from tensorpack.tfutils.variable import weight_create, bias_create
from tensorpack.proto.caffe_pb2 import FillerParameter

__all__ = ['Conv2D']


@layer.register()
def Conv2D(x, num_output, kernel_size, pad='SAME', stride=1, weight_filler=None, bias_filler=None,
           nl=tf.nn.relu, group=1, bias_term=True):
    """
    2D convolution on 4D inputs.

    :param input: a tensor of shape NHWC
    :param kernel_size: (h, w) or a int
    :param stride: (h, w) or a int. default to 1
    :param pad: 'valid' or 'same'. default to 'same'
    :param group: split channels as used in Alexnet. a int default to 1
    :param weight_filler:
    :param bias_filler: initializer for b. default to zero initializer.
    :param nl: nonlinearity. default to `relu`.
    :param bias_term: whether to use bias. a boolean default to True
    :returns: a NHWC tensor
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    assert in_channel % group == 0 and num_output % group == 0

    kernel_size = shape2d(kernel_size)
    pad = pad.upper()
    filter_shape = kernel_size + [in_channel / group, num_output]
    stride = shape4d(stride)
    if not weight_filler:
        weight_filler = FillerParameter(type='xavier')
    if bias_term and not bias_filler:
        bias_filler = FillerParameter(type='constant')
    w_var = weight_create(weight_filler, filter_shape)
    if bias_term:
        b_var = bias_create(bias_filler, [num_output])

    if group == 1:
        conv = tf.nn.conv2d(x, w_var, stride, pad)
    else:
        inputs = tf.split(3, group, x)
        kernels = tf.split(3, group, w_var)
        outputs = [tf.nn.conv2d(i, k, stride, pad)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(3, outputs)
    return nl(tf.nn.bias_add(conv, b_var) if bias_term else conv, name='output')


