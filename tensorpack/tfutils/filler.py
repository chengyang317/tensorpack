# -*- coding: UTF-8 -*-
# File: filler.py
# Author: philipcheng
# Time: 5/31/16 -> 5:46 PM
import tensorflow as tf
import numpy as np

__all__ = ['FillerConfig', 'get_initializer', 'get_w_default_initializer', 'get_b_default_initilizer']


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


def xavier_initializer(filter_shape, filler_config):
    """
    provie a xavier initializer
    :param filter_shape: h,w,ch_in,ch_out
    :param filler_config:
    :return:
    """
    variance_norm = filler_config.variance_norm.upper()
    assert variance_norm in ['AVERAGE', 'FAN_OUT']
    assert None not in filter_shape
    assert len(filter_shape) == 4

    fan_out = np.prod(filter_shape) / filter_shape[2]
    if variance_norm == 'AVERAGE':
        fan_in = np.prod(filter_shape[:3])
        n = (fan_in + fan_out) / 2
    else:
        n = fan_out
    scale = np.sqrt(3.0/n)
    return tf.random_uniform_initializer(-scale, scale)


def msra_initializer(filter_shape, filler_config):
    """
    provie a xavier initializer
    :param filter_shape: h,w,ch_in,ch_out
    :param filler_config:
    :return:
    """
    variance_norm = filler_config.variance_norm.upper()
    assert variance_norm in ['AVERAGE', 'FAN_IN']
    assert None not in filter_shape
    assert len(filter_shape) == 4

    fan_out = np.prod(filter_shape) / filter_shape[2]
    if variance_norm == 'AVERAGE':
        fan_in = np.prod(filter_shape[:3])
        n = (fan_in + fan_out) / 2
    else:
        n = fan_out
    std = np.sqrt(2.0/n)
    return tf.random_normal_initializer(stddev=std)


def constant_initializer(filter_shape, filler_config):
    """

    :param filter_shape:
    :param filler_config:
    :return:
    """
    value = filler_config.value
    return tf.constant_initializer(value)


def custom_initializer(filter_shape, filler_config):
    """

    :param filter_shape: h,w,ch_in,ch_out
    :param filler_config:
    :return:
    """
    value = np.array(filler_config.value)
    assert value.size == np.prod(filter_shape)
    value.reshape(filter_shape)
    return tf.constant_initializer(value)


def uniform_unit_scaling_initializer(filter_shape, filler_config):
    """

    :param filter_shape:
    :param filler_config:
    :return:
    """
    factor = filler_config.factor
    return tf.uniform_unit_scaling_initializer(factor=factor)


def random_norm_initializer(filter_shape, filler_config):
    """

    :param filter_shape:
    :param filler_config:
    :return:
    """
    mean = filler_config.mean
    std = filler_config.std
    dtype = filler_config.dtype
    seed = filler_config.seed
    return tf.random_normal_initializer(mean=mean, std=std, seed=seed, dtype=dtype)


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
        random_norm_initializer(filter_shape, filler_config)
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
