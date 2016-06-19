# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf

from ..utils import logger

__all__ = ['prediction_incorrect', 'one_hot', 'flatten', 'batch_flatten', 'logSoftmax', 'channel_flatten', 'cast_type',
           'class_balanced_binary_class_cross_entropy', 'print_stat', 'rms', 'get_shape', 'get_size', 'BOOLEN_TYPES',
           'NUMERIC_TYPES', 'mean_vector']

BOOLEN_TYPES = [tf.bool]
NUMERIC_TYPES = [tf.float32, tf.float16, tf.float16, tf.int32, tf.int64, tf.int8, tf.int16, tf.uint8, tf.uint16]

def one_hot(y, num_labels):
    """
    :param y: prediction. an Nx1 int tensor.
    :param num_labels: an int. number of output classes
    :returns: an NxC onehot matrix.
    """
    logger.warn("symbf.one_hot is deprecated in favor of more general tf.one_hot")
    return tf.one_hot(y, num_labels, 1.0, 0.0, dtype=tf.float32, name='one_hot')


def prediction_incorrect(logits, label, topk=1):
    """
    :param logits: NxC
    :param label: N
    :returns: a float32 vector of length N with 0/1 values, 1 meaning incorrect prediction
    """
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32)


def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, np.prod(shape)])
    return tf.reshape(x, tf.pack([tf.shape(x)[0], -1]))


def channel_flatten(x):
    """

    :param x:
    :return:
    """
    shape = x.get_shape().as_list()
    assert len(shape) == 4
    if shape[-1] is not None:
        return tf.reshape(x, [-1, shape[-1]])
    return tf.reshape(x, tf.pack([-1, tf.shape(x)[3]]))



def logSoftmax(x):
    """
    Batch log softmax.
    :param x: NxC tensor.
    :returns: NxC tensor.
    """
    logger.warn("symbf.logSoftmax is deprecated in favor of tf.nn.log_softmax")
    return tf.nn.log_softmax(x)


def class_balanced_binary_class_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss for binary classification,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    :param pred: size: b x ANYTHING. the predictions in [0,1].
    :param label: size: b x ANYTHING. the ground truth in {0,1}.
    :returns: class-balanced binary classification cross entropy loss
    """
    z = batch_flatten(pred)
    y = tf.cast(batch_flatten(label), tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    eps = 1e-8
    loss_pos = -beta * tf.reduce_mean(y * tf.log(tf.abs(z) + eps), 1)
    loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(tf.abs(1. - z) + eps), 1)
    cost = tf.sub(loss_pos, loss_neg)
    cost = tf.reduce_mean(cost, name=name)
    return cost


def print_stat(x, message=None):
    """ a simple print op.
        Use it like: x = print_stat(x)
    """
    if message is None:
        message = x.op.name
    return tf.Print(x, [tf.reduce_mean(x), x], summarize=20, message=message)


def rms(x, name=None):
    if name is None:
        name = x.op.name + '/rms'
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)


def get_shape(x):
    """
    get a shape of tensor
    :param x:
    :return:
    """
    shape = x.get_shape().as_list()
    if None not in shape:
        return shape
    shape_t = tf.shape(x)
    for i, item in enumerate(shape):
        if item is None:
            shape[i] = shape_t[i]
    return shape


def get_size(x):
    """

    :param x:
    :return:
    """
    shape = x.get_shape().as_list()
    if None not in shape:
        return np.prod(shape)
    return tf.size(x)


def cast_type(x, types):
    """
    Cast tensors x into a type belongs to types
    :param x: tensor or numpy
    :param types: type or list types
    :return:
    """
    if type(types) not in (tuple, list):
        types = [types]
    if x.dtype not in types:
        if type(x) is np.ndarray:
            return x.astype(types[0])
        else:
            return tf.cast(x, dtype=types[0])
    return x


def mean_vector(x):
    """
    Compute mean of a tensor x. It will omit nan value and inf value
    :param x: a vector tensor
    :return: a mean value tensor
    """
    x = flatten(x)
    x = cast_type(x, NUMERIC_TYPES)
    nan_mask = tf.is_nan()
    inf_mask = tf.is_inf()
    mask = tf.logical_or(nan_mask, inf_mask)
    x = tf.boolean_mask(x, mask=mask)
    return tf.reduce_mean(x)
