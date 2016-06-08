#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy

from ._common import *
from ..tfutils.symbolic_functions import *

__all__ = ['MaxPooling', 'FixedUnPooling', 'AvgPooling', 'GlobalAvgPooling',
           'BilinearUpSample']

@layer_register()
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

@layer_register()
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

@layer_register()
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

@layer_register()
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

@layer_register()
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


@layer_register()
def MaxPoolingWithArgmax(x, shape, stride=None, padding='VALID'):
    """
    MaxPooling on image and while return indices
    # this functon now only works on gpu!!!! Todo
    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor and indices tensor NHWC where store a flatten index
            ((b * height + y) * width + x) * channels + c.
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)
    return tf.nn.max_pool_with_argmax(x, ksize=shape, strides=stride, padding=padding)


def UnPooling2x2SameFilled(x):
    # out = tf.concat(3, [x, tf.identity(x)])
    # out = tf.concat(2, [out, tf.identity(out)])
    # sh = x.get_shape().as_list()
    # if None not in sh[1:]:
    #     out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    #     return tf.reshape(out, out_size)
    # else:
    #     sh = tf.shape(x)
    #     return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])
    return tf.tile(x, multiples=(1,2,2,1))


def unpooling2x2_argmax(x, argmax):
    unpool_x = UnPooling2x2SameFilled(x)




@layer_register()
def ArgmaxUnPooling(x, argmax, shape, stride=None):
    """
    Unpool the input x using indices tensor argmax which is from tensorflow max_pool_with_argmax.
    :param x:
    :param argmax:
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :return:
    """
    shape = shape2d(shape)
    if shape[0] == 2 and shape[1] == 2 and not stride:
        return unpooling2x2_argmax(x, argmax)
    raise LookupError('Fix it')



from ._test import TestModel
class TestPool(TestModel):
    def test_fixed_unpooling(self):
        h, w = 3, 4
        mat = np.random.rand(h, w, 3).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 3])
        output = FixedUnPooling('unpool', inp, 2)
        res = self.run_variable(output)
        self.assertEqual(res.shape, (1, 2*h, 2*w, 3))

        # mat is on cornser
        ele = res[0,::2,::2,0]
        self.assertTrue((ele == mat[:,:,0]).all())
        # the rest are zeros
        res[0,::2,::2,:] = 0
        self.assertTrue((res == 0).all())

    def test_upsample(self):
        h, w = 5, 5
        scale = 2

        mat = np.random.rand(h, w).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 1])

        output = BilinearUpSample('upsample', inp, scale)
        res = self.run_variable(output)

        from skimage.transform import rescale
        res2 = rescale(mat, scale)

        diff = np.abs(res2 - res[0,:,:,0])

        # not equivalent to rescale on edge
        diff[0,:] = 0
        diff[:,0] = 0
        if not diff.max() < 1e-4:
            import IPython;
            IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        self.assertTrue(diff.max() < 1e-4)


