# -*- coding: UTF-8 -*-
# File: predict_utils.py
# Author: philipcheng
# Time: 6/16/16 -> 10:04 AM
import tensorflow as tf
from tensorpack.tfutils.symbolic_functions import *

__all__ = ['SegmPredicts']


class SegmPredicts(object):
    """"""
    @staticmethod
    def pixel_accuracy(segm_eval, segm_gt):
        """

        :param segm_eval: (b,h,w,num_classes), logits
        :param segm_gt: (b,h,w)
        :return: (b,)
        """
        segm_eval_shape = tensor_shape(segm_eval)
        segm_gt_shape = tensor_shape(segm_gt)
        assert len(segm_eval_shape) == 4 and len(segm_gt_shape) == 3
        # shape is (b,h,w)
        segm_eval = tf.squeeze(tf.nn.top_k(segm_eval, k=1)[1])
        bool_true = tf.equal(segm_eval, segm_gt)
        # shape is (b,)
        accuracy = tf.reduce_mean(tf.cast(bool_true, tf.float32), reduction_indices=[1, 2])
        return accuracy

    @staticmethod
    def mean_accuracy(segm_eval, segm_gt):
        """

        :param segm_eval:(b,h,w,num_classes), logits
        :param segm_gt: (b,h,w)
        :return: (b,)
        """
        segm_eval_shape = tensor_shape(segm_eval)
        segm_gt_shape = tensor_shape(segm_gt)
        assert len(segm_eval_shape) == 4 and len(segm_gt_shape) == 3
        num_classes = segm_eval_shape[-1]
        # shape is (b,h,w,c)
        segm_eval = one_hot(tf.squeeze(tf.nn.top_k(segm_eval)[1]))
        segm_gt = one_hot(segm_gt, num_labels=num_classes)
        true_bool = tf.equal(segm_eval, segm_gt)
        # shape is (b,c)
        true_nums_per_class = tf.reduce_sum(tf.cast(true_bool, dtype=tf.float32), reduction_indices=[1, 2])
        gt_nums_per_class = tf.reduce_sum(segm_gt, reduction_indices=[1, 2])
        accuracy_per_class = tf.truediv(true_nums_per_class, gt_nums_per_class)

        accuracy_mean = tf.map_fn(lambda elem: mean_vector(elem), accuracy_per_class)
        return accuracy_mean

    @staticmethod
    def mean_IU(segm_eval, segm_gt):
        """

        :param segm_eval:
        :param segm_gt:
        :return: tensor of the mean_IU values, shape is (b,)
        """
        segm_eval_shape = tensor_shape(segm_eval)
        segm_gt_shape = tensor_shape(segm_gt)
        assert len(segm_eval_shape) == 4 and len(segm_gt_shape) == 3
        num_classes = segm_eval_shape[-1]
        # shape is (b,h,w,c)
        segm_eval = one_hot(tf.squeeze(tf.nn.top_k(segm_eval)[1]))
        segm_gt = one_hot(segm_gt, num_labels=num_classes)
        true_bool = tf.equal(segm_eval, segm_gt)
        # shape is (b,c)
        true_nums_per_class = tf.reduce_sum(tf.cast(true_bool, dtype=tf.float32), reduction_indices=[1, 2])
        gt_nums_per_class = tf.reduce_sum(segm_gt, reduction_indices=[1, 2])
        eval_nums_per_class = tf.reduce_sum(segm_eval, reduction_indices=[1, 2])
        IU_per_class = tf.truediv(true_nums_per_class, gt_nums_per_class + eval_nums_per_class - true_nums_per_class)

        accuracy_mean = tf.map_fn(lambda elem: mean_vector(elem), IU_per_class)
        return accuracy_mean

    @staticmethod
    def frequency_weighted_IU(segm_eval, segm_gt):
        """"""
        segm_eval_shape = tensor_shape(segm_eval)
        segm_gt_shape = tensor_shape(segm_gt)
        assert len(segm_eval_shape) == 4 and len(segm_gt_shape) == 3
        num_classes = segm_eval_shape[-1]
        # shape is (b,h,w,c)
        segm_eval = one_hot(tf.squeeze(tf.nn.top_k(segm_eval)[1]))
        segm_gt = one_hot(segm_gt, num_labels=num_classes)
        true_bool = tf.equal(segm_eval, segm_gt)
        # shape is (b,c)
        true_nums_per_class = tf.reduce_sum(tf.cast(true_bool, dtype=tf.float32), reduction_indices=[1, 2])
        gt_nums_per_class = tf.reduce_sum(segm_gt, reduction_indices=[1, 2])
        eval_nums_per_class = tf.reduce_sum(segm_eval, reduction_indices=[1, 2])
        IU_per_class = tf.truediv(true_nums_per_class, gt_nums_per_class + eval_nums_per_class - true_nums_per_class)
        frequency_per_class = tf.reduce_mean(gt_nums_per_class)
        fre_IU_per_class = IU_per_class * frequency_per_class

        accuracy_mean = tf.map_fn(lambda elem: mean_vector(elem), fre_IU_per_class)
        return accuracy_mean













