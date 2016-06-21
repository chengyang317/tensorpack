# -*- coding: UTF-8 -*-
# File: filler.py
# Author: philipcheng
# Time: 5/31/16 -> 5:46 PM
import tensorflow as tf
import numpy as np
from tensorpack.framework.registry import Registry
from tensorpack.proto import tensorpack_pb2 as pb

__all__ = ['FillerConfig', 'get_initializer', 'get_w_default_initializer', 'get_b_default_initilizer',
           'filler_registry']

filler_registry = Registry(register_name='filler')


class FillerConfig(object):
    """
    Config for initialize a filler
    """
    def __init__(self, **kwargs):
        self.type = kwargs.pop('type', 'random_normal')
        self.value = kwargs.pop('value', 0.0)
        self.min = kwargs.pop('min', 0.0)
        self.max = kwargs.pop('max', 1.0)
        self.mean = kwargs.pop('mean', 0.0)
        self.std = kwargs.pop('std', 1.0)
        self.factor = kwargs.pop('factor', 1.43)
        self.dtype = kwargs.pop('dtype', tf.float32)
        self.seed = kwargs.pop('dtype', None)
        self.variance_norm = kwargs.pop('variance_norm', 'FAN_IN')
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))


@filler_registry.register('xavier')
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
    if variance_norm == pb.FillerParameter.AVERAGE:
        n = (fan_in + fan_out) / 2
    elif variance_norm == pb.FillerParameter.FAN_OUT:
        n = fan_out
    scale = np.sqrt(3.0/n)
    return tf.random_uniform_initializer(-scale, scale)


@filler_registry.register('msra')
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
    if variance_norm == pb.FillerParameter.AVERAGE:
        n = (fan_in + fan_out) / 2
    elif variance_norm == pb.FillerParameter.FAN_OUT:
        n = fan_out
    std = np.sqrt(2.0/n)
    return tf.random_normal_initializer(stddev=std)


@filler_registry.register('constant')
def constant_initializer(filler_param, filler_shape=None):
    """
    :param filler_shape:
    :param filler_param:
    :return:
    """
    assert filler_param.type == 'constant'
    value = filler_param.value
    return tf.constant_initializer(value)


@filler_registry.register('custom')
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


@filler_registry.register('uniform_unit_scaling')
def uniform_unit_scaling_initializer(filler_param, filler_shape=None):
    """
    :param filler_param:
    :param filler_shape:
    :return:
    """
    assert filler_param.type == 'uniform_unit_scaling'
    factor = filler_param.factor
    return tf.uniform_unit_scaling_initializer(factor=factor)


@filler_registry.register('gaussian')
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


def get_initializer(filter_shape=None, filler_config=None):
    """
    Get a initializer fow w according the type of filler.
    :param filter_shape: h,w,ch_in,ch_out
    :param filler_config:
    :return:
    """
    if not filler_config:
        filler_config = FillerConfig(type='random_normal')
    if filler_config.type.upper() == 'XAVIER':
        return xavier_initializer(filter_shape, filler_config)
    elif filler_config.type.upper() == 'MSRA':
        return msra_initializer(filter_shape, filler_config)
    elif filler_config.type.upper() == 'CONSTANT':
        constant_initializer(filter_shape, filler_config)
    elif filler_config.type.upper() == 'CUSTOM':
        custom_initializer(filter_shape, filler_config)
    elif filler_config.type.upper() == 'RANDOM_NORM':
        gaussian_initializer(filter_shape, filler_config)
    elif filler_config.type.upper() == 'UNIFORM_UNIT_SCALING':
        uniform_unit_scaling_initializer(filter_shape, filler_config)
    else:
        raise KeyError('Please fix me!')


def get_w_default_initializer(filter_shape, filler_config=None):
    """
    Return a xavier initializer as default
    :param filter_shape:
    :param filler_config:
    :return:
    """
    assert not filler_config
    filler_config = FillerConfig(type='xavier', variance_norm='AVERAGE')
    get_initializer(filter_shape, filler_config)


def get_b_default_initilizer(filter_shape=None, filler_config=None):
    """

    :param filter_shape:
    :param filler_config:
    :return:
    """
    assert not filler_config
    filler_config = FillerConfig(type='constant')
    get_initializer(filter_shape, filler_config)



