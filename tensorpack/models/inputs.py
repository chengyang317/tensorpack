# -*- coding: UTF-8 -*-
# File: inputs.py
# Author: philipcheng
# Time: 6/21/16 -> 7:16 PM
import tensorflow as tf
from tensorpack.framework import Registry

__all__ = ['inputs_create_from_input']


def inputs_create_by_shape(input_params, input_shape_params):
    ret_inputs = {}
    for ind, input_name in enumerate(input_params):
        try:
            input_shape = input_shape_params[ind]
        except:
            input_shape = input_shape_params[0]
        ret_inputs[input_name] = tf.placeholder(dtype=tf.float32, shape=input_shape.dim, name=input_name)
    return ret_inputs


def inputs_create_by_dim(input_params, input_dim_params):
    ret_inputs = {}
    assert len(input_dim_params) == 4 * len(input_params)
    for ind, input_name in enumerate(input_params):
        shape = input_dim_params[ind * 4:ind * 4 + 4]
        ret_inputs[input_name] = tf.placeholder(dtype=tf.float32, shape=shape, name=input_name)
    return ret_inputs


def inputs_create_from_input(net_params):
    if net_params.HasField('input_shape'):
        inputs_create_by_shape(net_params.input, net_params.input_shape)
    elif net_params.HasField('input_dim'):
        inputs_create_by_dim(net_params.input, net_params.input_dim)
    else:
        raise EnvironmentError("input_dim or input_shape must exits one")


def inputs_create_from_layer(net_params):
    data_params = net_params.ListFields()
