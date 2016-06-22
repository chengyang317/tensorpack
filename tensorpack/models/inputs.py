# -*- coding: UTF-8 -*-
# File: inputs.py
# Author: philipcheng
# Time: 6/21/16 -> 7:16 PM
import tensorflow as tf
from tensorpack.framework import Registry

__all__ = ['inputs_create']

inputs_create_func_register = Registry(register_name='inputs_create_func')


@inputs_create_func_register.register(key='BlobShape')
def inputs_create_by_blobshape(input_params, input_shape_params):
    ret_inputs = {}
    for ind, input_name in enumerate(input_params):
        try:
            input_blob_shape = input_shape_params[ind]
        except:
            input_blob_shape = input_shape_params[0]
        ret_inputs[input_name] = tf.placeholder(dtype=tf.float32, shape=input_blob_shape.dim, name=input_name)
    return ret_inputs


@inputs_create_func_register.register(key='dim')
def inputs_create_by_dim(input_params, input_shape_params):
    ret_inputs = {}
    for ind, input_name in enumerate(input_params):
        shape = input_shape_params[ind*4:ind*4+4]
        ret_inputs[input_name] = tf.placeholder(dtype=tf.float32, shape=shape, name=input_name)
    return ret_inputs


def inputs_create(input_params, input_shape_params, input_shape_params_type):
    inputs_create_func = inputs_create_func_register.look_up(input_shape_params_type)
    return inputs_create_func(input_params, input_shape_params)
