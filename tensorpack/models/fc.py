#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorpack.tfutils.filler import *
from tensorpack.models.utils import layer_manage
from tensorpack.tfutils.symbolic_functions import *

__all__ = ['FullyConnected']


@layer_manage.register()
def FullyConnected(x, out_dim, w_init_config=None, b_init_config=None, nl=tf.nn.relu, use_bias=True):
    """
    Fully-Connected layer.

    :param input: a tensor to be flattened except the first dimension.
    :param out_dim: output dimension
    :param w_init_config: initializer for W. default to `xavier_initializer_conv2d`.
    :param b_init_config: initializer for b. default to zero initializer.
    :param nl: nonlinearity. default to `relu`.
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a 2D tensor
    """
    x = batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]
    filter_shape = [in_dim, out_dim]
    if not w_init_config:
        w_init_config = FillerConfig(type='uniform_unit_scaling', factor=1.43)
    if not b_init_config:
        b_init_config = FillerConfig(type='constant', value=0.0)
    w_initializer = get_initializer(filter_shape, w_init_config)
    b_initializer = get_initializer([out_dim], b_init_config)
    W = tf.get_variable('W', [in_dim, out_dim], initializer=w_initializer)
    if use_bias:
        b = tf.get_variable('b', [out_dim], initializer=b_initializer)
    prod = tf.nn.xw_plus_b(x, W, b) if use_bias else tf.matmul(x, W)
    return nl(prod, name='output')
