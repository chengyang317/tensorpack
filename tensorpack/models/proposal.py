#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: proposal
# Author: Philip Cheng
# Time: 6/29/16 -> 4:31 PM
from tensorpack.tools.anchors import generate_anchors, anchors_shift
from tensorpack.tfutils.symbolic_functions import tensor_shape
from tensorpack.tfutils.object_detection.bbox_transform import *


def Proposal(inputs, feat_strid, im_info, base_size=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """
    :param inputs: (cls_prob, bbox_prob) shape is (b,h,w,2*anchor_nums), (b,h,w,4*anchor_nums)
    :param feat_strid: the ratio of origin_size/feature_size
    :param im_info: input_image information like (orgin_h,origin_w,scale)
    :param base_size: anchor base size, default is 16
    :param anchor_scales:
    :param anchor_ratios:
    :return:
    """
    assert len(inputs) == 2
    cls_prob, bbox_prob = inputs
    cls_prob_shape = tensor_shape(cls_prob)
    bbox_prob_shape = tensor_shape(bbox_prob)
    anchor_nums = len(anchor_ratios) * len(anchor_scales)
    assert cls_prob_shape[-1] == 2 * anchor_nums
    assert bbox_prob_shape == 4 * anchor_nums

    height, width = cls_prob_shape[1:3]
    # create shift anchors whose shape should be (h,w,anchor_nums,4)
    anchors = generate_anchors(base_size=base_size, ratios=anchor_ratios, scales=anchor_scales)
    shift_anchors = anchors_shift(anchors, width, height, feat_strid)
    # According anchors and bbox_prob to create proposals
    proposals = proposal_from_delta(boxes=anchors, bbox_deltas=bbox_prob)
    proposals = clip_boxes(proposals, im_info[:2])
    keep = filter_boxes(proposals, min_size * im_info[2])




















