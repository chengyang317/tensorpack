# -*- coding: UTF-8 -*-
# File: transformation.py
# Author: philipcheng
# Time: 6/10/16 -> 5:06 PM
import tensorflow as tf

__all__ = ['tensor_repeat', 'tensor_repeats']


def tensor_repeat(x, dims, ntimes):
    """
    repeat tensor x in dims dimensions for ntimes, if dim has not been exited, then expand it
    :param x: tensor
    :param dims: int or list
    :param ntimes: int or tensorshape[i] or list
    :return: new tensor
    """
    if type(dims) is not list:
        dims = [dims]
    if type(ntimes) is not list:
        ntimes = [ntimes]
    dim_dict = {}
    for dim, ntime in zip(dims, ntimes):
        dim_dict[dim] = ntime
    for dim in dims:
        while x.get_shape().ndims < dim + 1:
            x = tf.expand_dims(x, -1)
    multiples = []
    for i in xrange(x.get_shape().ndims):
        if i in dims:
            multiples.append(dim_dict[i])
        else:
            multiples.append(1)
    return tf.tile(x, multiples=multiples)


def tensor_repeats(x, multiples, axis=0):
    """

    :param x:
    :param multiples:
    :param axis:
    :return:
    """
    new_dims = len(multiples)
    x_dims = x.get_shape().ndims
    if axis == 0:
        while x.get_shape().ndims < new_dims:
            x = tf.expand_dims(x, -1)
    elif axis == 1:
        while x.get_shape().ndims < new_dims:
            x = tf.expand_dims(x, 0)
    else:
        raise ValueError
    return tf.tile(x, multiples)


