# -*- coding: UTF-8 -*-
# File: accuracy.py
# Author: philipcheng
# Time: 6/17/16 -> 6:20 PM
import tensorflow as tf
from tensorpack.predict.predict_utils import SegmPredicts
from tensorpack.models.utils import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.naming import *

__all__ = ['segm_pixel_accuracy', 'segm_mean_accuracy', 'segm_mean_IU', 'segm_frequency_weighted_IU',
           'classification_accuracy']


@layer_manage.register(log_shape=False)
def segm_pixel_accuracy(segm_eval, segm_gt, keys=None):
    """
    Segmentation pixel accuracy layer
    :param segm_eval: it is logits and shape is (b,h,w,c)
    :param segm_gt: it is groundtruth label and shape is (b,h,w)
    :param keys: add the output to graph keys's collection
    :return: a tensor whose shape is (b,)
    """
    accuracy = SegmPredicts.pixel_accuracy(segm_eval, segm_gt)
    ret = tf.identity(accuracy, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer_manage.register()
def segm_mean_accuracy(segm_eval, segm_gt, keys=None):
    """

    :param segm_eval:
    :param segm_gt:
    :return:
    """
    accuracy = SegmPredicts.mean_accuracy(segm_eval, segm_gt)
    ret = tf.identity(accuracy, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer_manage.register()
def segm_mean_IU(segm_eval, segm_gt, keys=None):
    """

    :param segm_eval:
    :param segm_gt:
    :return:
    """
    accuracy = SegmPredicts.mean_IU(segm_eval, segm_gt)
    ret = tf.identity(accuracy, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer_manage.register()
def segm_frequency_weighted_IU(segm_eval, segm_gt, keys=None):
    """

    :param segm_eval:
    :param segm_gt:
    :return:
    """
    accuracy = SegmPredicts.frequency_weighted_IU(segm_eval, segm_gt)
    ret = tf.identity(accuracy, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


@layer_manage.register()
def classification_accuracy(logits, labels, right=False, keys=None):
    if right:
        correct = prediction_correct(logits, labels)
    correct = prediction_incorrect(logits, labels)
    ret = tf.reduce_sum(correct, name='output')
    if keys is not None:
        cast_list(keys)
        for key in keys:
            tf.add_to_collection(key, ret)
    return ret


