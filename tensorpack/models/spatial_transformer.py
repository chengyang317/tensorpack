# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: philipcheng
# Time: 6/7/16 -> 11:55 PM
import tensorflow as tf
from tensorpack.models._common import layer_register
from tensorpack.tfutils.meshgrid import meshgrid_3d
from tensorpack.tfutils.transformation import *
import numpy as np

__all__ = ['spatial_transformer']


def interpolate(x, x_cor, y_cor, target_shape):
    """
    input matrix is im, return a output tensors (b,oh,ow) whose value is from im matrix
    :param x: input tensor with dimension (b*h*w*c), only b can be unsure.
    :param x_cor: tensor about map relation with dimension (b,prod(target_shape))
    :param y_cor: (b,prod(target_shape))
    :param target_shape: target 2d shape (h,w)
    :return:
    """
    assert type(target_shape) in (list, tuple) and len(target_shape) == 2
    # infer the shape as soon as possbile
    x_shape = x.get_shape().as_list()
    if not x_shape[0]:
        x_shape[0] = tf.shape(x)[0]
    x_cor_shape = x_cor.get_shape().as_list()
    y_cor_shape = y_cor.get_shape().as_list()
    assert len(x_shape) == 4 and len(x_cor_shape) == 2 and len(y_cor_shape) == 2
    x_max_height = x_shape[1]
    x_max_width = x_shape[2]
    # scale indices from [-1, 1] to [0, width/height]
    if x_cor.dtype != tf.float32 or y_cor.dtype != tf.float32:
        x_cor = tf.cast(x_cor, tf.float32)
        y_cor = tf.cast(y_cor, tf.float32)
    # adjust the shape of s_x to (b, target_shape[0], target_shape[1])
    x_cor = tf.reshape(x_cor, (-1, target_shape[0], target_shape[1]))
    y_cor = tf.reshape(y_cor, (-1, target_shape[0], target_shape[1]))
    x_cor = (x_cor + 1.0) * x_shape[2] / 2.0
    y_cor = (y_cor + 1.0) * x_shape[1] / 2.0
    x0_cor = tf.cast(tf.floor(x_cor), 'int32')
    x1_cor = x0_cor + 1
    y0_cor = tf.cast(tf.floor(y_cor), 'int32')
    y1_cor = y0_cor + 1
    x0_cor = tf.clip_by_value(x0_cor, 0, x_max_height)
    x1_cor = tf.clip_by_value(x1_cor, 0, x_max_height)
    y0_cor = tf.clip_by_value(y0_cor, 0, x_max_width)
    y1_cor = tf.clip_by_value(y1_cor, 0, x_max_width)
    # construct an indices base matrix (b*target_shape[0]*target_shape[1])
    indices_base = tensor_repeat(tf.range(x_shape[0]) * x_shape[1] * x_shape[2],
                                 [1, 2], [target_shape[0], target_shape[1]])
    # construct four indices whose dimension is (b, oh, ow), every item value is an index in the matrix im.
    indices_y0_base = indices_base + y0_cor * x_shape[2]
    indices_y1_base = indices_base + y1_cor * x_shape[2]
    indices_a = indices_y0_base + x0_cor
    indices_b = indices_y1_base + x0_cor
    indices_c = indices_y0_base + x1_cor
    indices_d = indices_y1_base + x1_cor
    # adjust the im to the shape (-1, channels)
    im_flat = tf.reshape(x, [-1, x_shape[3]])
    # use tf.gather to get value from im matrix for every item in target matrix
    # output[i, ..., j, :, ...:] = params[indices[i, ..., j], :, ..., :]
    target_a = tf.gather(params=im_flat, indices=indices_a)
    target_b = tf.gather(params=im_flat, indices=indices_b)
    target_c = tf.gather(params=im_flat, indices=indices_c)
    target_d = tf.gather(params=im_flat, indices=indices_d)
    # comput weights and the finally calculate the interpolated values
    w_a = (tf.cast(x1_cor, tf.float32) - x_cor) * (tf.cast(y1_cor, tf.float32) - y_cor)
    w_b = (tf.cast(x1_cor, tf.float32) - x_cor) * (y_cor - tf.cast(y0_cor, tf.float32))
    w_c = (x_cor - tf.cast(x0_cor, tf.float32)) * (tf.cast(y1_cor, tf.float32) - y_cor)
    w_d = (x_cor - tf.cast(x0_cor, tf.float32)) * (y_cor - tf.cast(y0_cor, tf.float32))
    w_a = tensor_repeat(w_a, 3, x_shape[3])
    w_b = tensor_repeat(w_b, 3, x_shape[3])
    w_c = tensor_repeat(w_c, 3, x_shape[3])
    w_d = tensor_repeat(w_d, 3, x_shape[3])
    return tf.add_n([w_a*target_a, w_b*target_b, w_c*target_c, w_d*target_d])


def transformer(x, theta, out_shape):
    """

    :param x:
    :param theta:
    :param out_shape: (h,w)
    :return:
    """
    assert type(out_shape) in (list, tuple) and len(out_shape) == 2
    # adjust theta to (b,2,3)
    theta = tf.cast(tf.reshape(theta, (-1, 2, 3)), tf.float32)
    x_shape = tf.shape(x)
    mesh_grid = meshgrid_3d(x_shape[0], out_shape[0], out_shape[1])
    # (b, 2, oh*ow)
    xy_cor = tf.batch_matmul(theta, mesh_grid)
    x_cor = tf.squeeze(tf.slice(xy_cor, [0, 0, 0], [-1, 1, -1]), [1])
    y_cor = tf.squeeze(tf.slice(xy_cor, [0, 1, 0], [-1, 1, -1]), [1])
    out = interpolate(x, x_cor, y_cor, out_shape)
    return out


@layer_register()
def spatial_transformer(x, local_net, out_shape):
    """

    :param x:
    :param local_net:
    :param out_shape: h,w
    :return:
    """
    assert type(out_shape) in (list, tuple) and len(out_shape) == 2
    theta = local_net(x)
    # theta shape must be (b,6)
    assert theta.get_shape().ndims == 2 and theta.get_shape().as_list()[-1] == 6
    return transformer(x, theta, out_shape)
