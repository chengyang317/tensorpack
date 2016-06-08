# -*- coding: UTF-8 -*-
# File: meshgrid.py
# Author: philipcheng
# Time: 6/7/16 -> 4:23 PM
import tensorflow as tf

__all__ = ['meshgrid_2d', 'meshgrid_3d']


def meshgrid_2d(height, width):
    """
    Create a meshgrid for 2D matrix
    :param height:
    :param width:
    :return: (3,width*height)
    """
    x_c = tf.expand_dims(tf.lin_space(-1.0, 1.0, width), 0) #(1,width)
    x_t = tf.tile(x_c, (height, 1))  #(height, width)
    y_c = tf.expand_dims(tf.lin_space(-1.0, 1.0, height), -1)
    y_t = tf.tile(y_c, (1, width)) #(height, width)
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    ones = tf.ones_like(x_t_flat)
    grid = tf.concat(0, [x_t_flat, y_t_flat, ones]) #(3,width*height)
    return grid


def meshgrid_3d(batch, height, width):
    """

    :param batch:
    :param height:
    :param width:
    :return: (b, 3, width*height
    """
    # (3, width*height)
    grid_2d = meshgrid_2d(height, width)
    grid = tf.expand_dims(grid_2d, 0)
    return tf.tile(grid, (batch, 1, 1))




