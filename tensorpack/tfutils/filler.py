# -*- coding: UTF-8 -*-
# File: filler.py
# Author: philipcheng
# Time: 5/31/16 -> 5:46 PM
import tensorflow as tf
import numpy as np
from tensorpack.framework.registry import Registry
from tensorpack.proto.caffe_pb2 import FillerParameter

__all__ = ['filler_initializer_register']

filler_initializer_register = Registry(register_name='filler')


@filler_initializer_register.register('xavier')
def xavier_initializer(filler_param, filler_shape):
    """
    provie a xavier initializer
    :param filler_shape: h,w,ch_in,ch_out
    :param filler_param:
    :return:
    """
    assert filler_param.type == 'xavier'
    variance_norm = filler_param.variance_norm
    fan_out = np.prod(filler_shape) / filler_shape[2]
    fan_in = np.prod(filler_shape[:3])
    n = fan_in  # default is fan_in
    if variance_norm == FillerParameter.AVERAGE:
        n = (fan_in + fan_out) / 2
    elif variance_norm == FillerParameter.FAN_OUT:
        n = fan_out
    scale = np.sqrt(3.0/n)
    return tf.random_uniform_initializer(-scale, scale)


@filler_initializer_register.register('msra')
def msra_initializer(filler_param, filler_shape):
    """
    provie a xavier initializer
    :param filler_shape: h,w,ch_in,ch_out
    :param filler_param:
    :return:
    """
    assert filler_param.type == 'msra'
    variance_norm = filler_param.variance_norm
    fan_out = np.prod(filler_shape) / filler_shape[2]
    fan_in = np.prod(filler_shape[:3])
    n = fan_in  # default is fan_in
    if variance_norm == FillerParameter.AVERAGE:
        n = (fan_in + fan_out) / 2
    elif variance_norm == FillerParameter.FAN_OUT:
        n = fan_out
    std = np.sqrt(2.0/n)
    return tf.random_normal_initializer(stddev=std)


@filler_initializer_register.register('constant')
def constant_initializer(filler_param, filler_shape=None):
    """
    :param filler_shape:
    :param filler_param:
    :return:
    """
    assert filler_param.type == 'constant'
    value = filler_param.value
    return tf.constant_initializer(value)


@filler_initializer_register.register('custom')
def custom_initializer(filler_param, filler_shape):
    """
    :param filler_param: h,w,ch_in,ch_out
    :param filler_shape:
    :return:
    """
    assert filler_param.type == 'custom'
    custom_value = np.array(filler_param.custom_value)
    assert custom_value.size == np.prod(filler_shape)
    custom_value.reshape(filler_shape)
    return tf.constant_initializer(custom_value)


@filler_initializer_register.register('uniform_unit_scaling')
def uniform_unit_scaling_initializer(filler_param, filler_shape=None):
    """
    :param filler_param:
    :param filler_shape:
    :return:
    """
    assert filler_param.type == 'uniform_unit_scaling'
    factor = filler_param.factor
    return tf.uniform_unit_scaling_initializer(factor=factor)


@filler_initializer_register.register('gaussian')
def gaussian_initializer(filler_param, filler_shape=None):
    """
    :param filler_param:
    :param filler_shape:
    :return:
    """
    assert filler_param.type == 'gaussian'
    mean = filler_shape.mean
    std = filler_shape.std
    return tf.random_normal_initializer(mean=mean, stddev=std)




