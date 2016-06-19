# -*- coding: UTF-8 -*-
# File: loss.py
# Author: philipcheng
# Time: 6/16/16 -> 9:56 AM
import tensorflow as tf
from tensorpack.models.utils import layer
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.utils import memoized
from tensorpack.utils import logger
import re

__all__ = ['segm_loss', 'regularize_loss', 'classification_loss', 'sum_loss']


@layer.register()
def classification_loss(logits, labels, sparse=True, keys=None):
    logits = cast_type(logits, [tf.float32, tf.float64])
    logits = channel_flatten(logits)
    if sparse:
        labels = cast_type(labels, [tf.int32, tf.int64])
        labels = flatten(labels)
        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    else:
        labels = cast_like(labels, logits)
        labels = reshape_like(labels, logits)
        cross_loss = tf.nn.softmax_cross_entropy_with_logits
    ret = tf.reduce_mean(cross_loss, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer.register()
def segm_loss(segm_eval, segm_gt, keys=None):
    """

    :param segm_eval: (b,h,w,c) segementation eval image and it's logits with out soft_max
    :param segm_gt: groud truth image. (b,h,w)
    :return:
    """
    segm_eval_shape = get_shape(segm_eval)
    segm_gt_shape = get_shape(segm_gt)
    assert len(segm_eval_shape) == 4 and len(segm_gt_shape) == 3
    # reshape the segm_eval to (b*h*w,c), segm_gt to (b*h*w,)
    logits = tf.reshape(segm_eval, [-1, segm_eval_shape[-1]])
    labels = tf.reshape(segm_gt, [-1])
    cast_type(logits, [tf.float32, tf.float64])
    cast_type(labels, [tf.int32, tf.int64])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    ret = tf.reduce_mean(loss, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))


@layer.register()
def regularize_loss(regex, weight_delay=0.004, loss_func=tf.nn.l2_loss, keys=None):
    """
        Apply a regularizer on every trainable variable matching the regex.
        :param regex: regex template used to match variables
        :param weight_delay:
        :param loss_func: a function that takes a tensor and return a scalar.
        """
    G = tf.get_default_graph()
    variables = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    losses = []
    for v in variables:
        v_name = v.name
        if re.search(regex, v_name):
            losses.append(loss_func(v))
            _log_regularizer(v_name)
    if not losses:
        raise IndexError("can't mach regex {}".format(regex))
    loss = tf.add_n(losses)
    ret = tf.mul(weight_delay, loss, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer.register()
def sum_loss(losses, keys=None):
    ret = tf.add_n(losses, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret



