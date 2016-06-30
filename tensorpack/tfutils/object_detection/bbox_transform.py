#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: bbox_transform
# Author: Philip Cheng
# Time: 6/30/16 -> 10:24 AM
import numpy as np
import tensorflow as tf
from tensorpack.tfutils.symbolic_functions import tensor_shape, cast_type

__all__ = ['proposal_from_delta', "clip_boxes", "filter_boxes"]


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def proposal_from_delta(boxes, bbox_deltas):
    """
    :param boxes: (h,w, anchor_nums, 4), type is ndarray
    :param bbox_deltas: (b,h,w,4*anchor_nums), type is tenosor
    :return:
    """
    bbox_shape = tensor_shape(bbox_deltas)
    # convert shape of boxes to (b*h*w*anchor_nums, 4)
    boxes = tf.reshape(tf.tile(tf.convert_to_tensor(boxes), multiples=(bbox_shape[0], 1, 1, 1, 1)), shape=(-1, 4))
    # Calculate the w,h,central_x, central_y
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    # convet shape of bbox_deltas to (b*h*w*anchor_nums, 4)
    bbox_deltas = tf.reshape(bbox_deltas, shape=(-1, 4))
    # Calculate dx,dy,dw,dh
    dx = bbox_deltas[:, 0]
    dy = bbox_deltas[:, 1]
    dw = bbox_deltas[:, 2]
    dh = bbox_deltas[:, 3]
    # Calculate the predicted boxes attrs
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights
    # Construct the boxes according to attrs
    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
    pred_boxes = tf.pack((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2))
    pred_boxes = tf.reshape(pred_boxes, shape=bbox_shape)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: (b,h,w,anchor_nums*4)
    :param im_shape: (origin_h,origin_w)
    """
    boxes = cast_type(tf.reshape(boxes, (-1, 4)), tf.int32)
    # convert to be (w,h,w,h)
    im_shape = tf.convert_to_tensor(im_shape[::-1]*2, dtype=tf.int32)
    boxes = tf.maximum(tf.minimum(boxes, im_shape), tf.zeros(shape=(4,), dtype=tf.int32))
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    boxes = cast_type(tf.reshape(boxes, (-1, 4)), tf.int32)
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = (ws >= min_size) & (hs >= min_size)
    return keep
