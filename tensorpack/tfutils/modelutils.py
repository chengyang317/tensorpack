# -*- coding: UTF-8 -*-
# File: modelutils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
from tensorpack.utils import logger

__all__ = ['describe_model', 'get_shape_str', 'get_name_str']


def describe_model():
    """ print a description of the current model parameters """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    msg = [""]
    total = 0
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        msg.append("{}: shape={}, dim={}".format(
            v.name, shape.as_list(), ele))
    msg.append("Total dim={}".format(total))
    logger.info("Model Params: {}".format('\n'.join(msg)))


def get_shape_str(tensors):
    """
    :param tensors: a tensor or a list of tensors
    :returns: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str


def get_name_str(graph_elements):
    """
    :param graph_elements: list of tensors, variables, ops
    :return: a string to describe their name
    """
    if not isinstance(graph_elements, (list, tuple)):
        graph_elements = list(graph_elements)
    if not graph_elements:
        return 'None'
    name_strs = [elem.name for elem in graph_elements]
    return ','.join(name_strs)
