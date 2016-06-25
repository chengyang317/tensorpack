#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
from tensorpack.proto.caffe_pb2 import FillerParameter
from tensorpack.models.utils import layer
from tensorpack.tfutils.symbolic_functions import batch_flatten
from tensorpack.tfutils.variable import weight_create, bias_create

__all__ = ['FullyConnected']


@layer.register()
def FullyConnected(x, num_output, weight_filler=None, bias_filler=None, nl=tf.nn.relu, bias_term=True):
    """
    Fully-Connected layer.

    :param input: a tensor to be flattened except the first dimension.
    :param num_output: output dimension
    :param weight_filler: initializer for W. default to `xavier_initializer_conv2d`.
    :param bias_filler: initializer for b. default to zero initializer.
    :param nl: nonlinearity. default to `relu`.
    :param bias_term: whether to use bias. a boolean default to True
    :returns: a 2D tensor
    """
    x = batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]
    filter_shape = [in_dim, num_output]
    if not weight_filler:
        weight_filler = FillerParameter(type='uniform_unit_scaling', factor=1.43)
    if bias_term and not bias_filler:
        bias_filler = FillerParameter(type='constant')
    w_var = weight_create(weight_filler, filter_shape)
    if bias_term:
        b_var = bias_create(bias_filler, [num_output])
    prod = tf.nn.xw_plus_b(x, w_var, b_var) if bias_term else tf.matmul(x, w_var)
    return nl(prod, name='output')

