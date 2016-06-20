#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy as np
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.models.utils import *
from tensorpack.tfutils.transformation import tensor_repeats

__all__ = ['MaxPooling', 'FixedUnPooling', 'AvgPooling', 'GlobalAvgPooling', 'BilinearUpSample',
           'MaxPoolingWithArgmax', 'ArgmaxUnPooling']

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, some_other_arg):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')


@layer.register()
def MaxPooling(x, shape, stride=None, padding='VALID'):
    """
    MaxPooling on images.

    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor.
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


@layer.register()
def AvgPooling(x, shape, stride=None, padding='VALID'):
    """
    Average pooling on images.

    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor.
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.avg_pool(x, ksize=shape, strides=stride, padding=padding)


@layer.register()
def GlobalAvgPooling(x):
    """
    Global average pooling as in `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.

    :param input: NHWC tensor.
    :returns: NC tensor.
    """
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


# https://github.com/tensorflow/tensorflow/issues/2169
def UnPooling2x2ZeroFilled(x):
    out = tf.concat(3, [x, tf.zeros_like(x)])
    out = tf.concat(2, [out, tf.zeros_like(out)])

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        sh = tf.shape(x)
        return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


@layer.register()
def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed mat to perform kronecker product with.

    :param input: NHWC tensor
    :param shape: int or [h, w]
    :param unpool_mat: a tf/np matrix with size=shape. If None, will use a mat
        with 1 at top-left corner.
    :returns: NHWC tensor
    """
    shape = shape2d(shape)

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)

    input_shape = tf.shape(x)
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.Variable(mat, trainable=False, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.Variable(unpool_mat, trainable=False, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(flatten(unpool_mat), 0)    #1x(shxsw)
    prod = tf.matmul(fx, mat)    #(bchw) x(shxsw)
    prod = tf.reshape(prod, tf.pack(
        [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, tf.pack(
        [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
    return prod


@layer.register()
def BilinearUpSample(x, shape):
    """
    Bilinear upsample the input images.
    :param x: input NHWC tensor
    :param shape: an integer
    """
    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        See https://github.com/BVLC/caffe/blob/master/include%2Fcaffe%2Ffiller.hpp#L244
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x,y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret

    ch = x.get_shape().as_list()[3]
    shape = int(shape)
    unpool_mat = np.zeros((shape, shape), dtype='float32')
    unpool_mat[-1,-1] = 1
    x = FixedUnPooling('unpool', x, shape, unpool_mat)

    filter_shape = 2 * shape
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))
    weight_var = tf.constant(w,
                             tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch))

    output = tf.nn.conv2d(x, weight_var, [1,1,1,1], padding='SAME')
    return output


@layer.register()
def MaxPoolingWithArgmax(x, shape, stride=None, padding='VALID'):
    """
    MaxPooling on image and while return indices
    # this functon now only works on gpu!!!! Todo
    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor and indices tensor NHWC where store a flatten index
            ((b * height + y) * width + x) * channels + c.  ###### not true, the experiment shows the index
            does not involved with b. just as (y * width + x) * channels + c for every batch
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)
    return tf.nn.max_pool_with_argmax(x, ksize=shape, strides=stride, padding=padding)


def unpooling2x2_argmax(x, argmax):
    """

    :param x: (b,h,w,c)
    :param argmax: (b,h,w,c), (y * width + x) * channels + c.
    :return:
    """
    # assert x.get_shape().as_list() == argmax.get_shape().as_list()
    # infer the shape as soon as possbile
    x_shape = x.get_shape().as_list()
    if not x_shape[0]:
        x_shape[0] = tf.shape(x)[0]
    unpool_x = tf.depth_to_space(tf.tile(x, multiples=(1, 1, 1, 4)), block_size=2)
    # chang argmax value from flatten index to be in range(4) for the index for a localization
    channel_base = tf.cast(tensor_repeats(tf.range(x_shape[3]), [x_shape[0], x_shape[1], x_shape[2], 1], axis=1),
                           dtype=tf.int64)
    # y * width + x
    argmax = tf.div(argmax - channel_base, x_shape[3])
    # x
    argmax_x_ind = tf.mod(argmax, x_shape[2] * 2)
    # y
    argmax_y_ind = tf.div(argmax, x_shape[2] * 2)
    # ajust x, y to range(2)
    argmax_x_ind = tf.mod(argmax_x_ind, 2)
    argmax_y_ind = tf.mod(argmax_y_ind, 2)
    argmax_ind = argmax_y_ind * 2 + argmax_x_ind
    # change argmax to (b*c,h,w)
    argmax_ind = tf.transpose(argmax_ind, [0, 3, 1, 2])
    argmax_ind = tf.reshape(argmax_ind, [-1, x_shape[1], x_shape[2]])
    # (b*c,h,w,depth)
    template = tf.one_hot(indices=argmax_ind, depth=4, dtype=tf.float32)
    template = tf.squeeze(tf.depth_to_space(template, block_size=2))
    template = tf.reshape(template, [-1, x_shape[3], x_shape[1]*2, x_shape[2]*2])
    template = tf.transpose(template, [0, 2, 3, 1])
    return tf.mul(unpool_x, template)


@layer.register()
def ArgmaxUnPooling(x, argmax, shape, stride=None):
    """
    Unpool the input x using indices tensor argmax which is from tensorflow max_pool_with_argmax.
    :param x: (b,h,w,c)
    :param argmax: (b,h,w,c)
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :return:
    """
    shape = shape2d(shape)
    stride = shape2d(stride)
    if shape[0] == 2 and shape[1] == 2 and stride[0] == 2 and stride[1] == 2:
        return unpooling2x2_argmax(x, argmax)
    raise LookupError('Fix it')






