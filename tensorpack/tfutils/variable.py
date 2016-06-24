# -*- coding: UTF-8 -*-
# File: variable.py
# Author: philipcheng
# Time: 6/20/16 -> 10:00 PM
import tensorflow as tf
from tensorpack.proto import tensorpack_pb2 as pb
from tensorpack.tfutils.filler import filler_initializer_register

__all__ = ['variable_create', 'weight_create', 'bias_create']


def get_initializer(filler_param, var_shape):
    initializer_func = filler_initializer_register.look_up(filler_param.type)
    var_initializer = initializer_func(filler_param, var_shape)
    return var_initializer


def variable_create(filler_param, var_shape, var_name=None):
    assert isinstance(var_shape, (tuple, list))
    assert isinstance(filler_param, pb.FillerParameter)
    var_initializer = get_initializer(filler_param, var_shape)
    return tf.get_variable(var_name, var_shape, initializer=var_initializer)


def weight_create(filler_param, var_shape):
    var_name = 'weight'
    return variable_create(filler_param, var_shape, var_name)


def bias_create(filler_param, var_shape):
    var_name = 'bias'
    return variable_create(filler_param, var_shape, var_name)










