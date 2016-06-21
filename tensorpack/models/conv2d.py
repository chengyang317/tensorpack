#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
from tensorpack.tfutils.filler import *
from tensorpack.models.utils import layer_manage
from tensorpack.models.layer import Layer
from tensorpack.tfutils.symbolic_functions import cast_list, tensor_shape
from tensorpack.tfutils.variable import WeightCreater, BisaCreater
from tensorpack.proto import tensorpack_pb2 as pb

__all__ = ['Conv2D']


@layer_manage.register()
def Conv2D(x, out_channel, kernel_shape, padding='SAME', stride=1, w_init_config=None, b_init_config=None,
           nl=tf.nn.relu, split=1, use_bias=True):
    """
    2D convolution on 4D inputs.

    :param input: a tensor of shape NHWC
    :param kernel_shape: (h, w) or a int
    :param stride: (h, w) or a int. default to 1
    :param padding: 'valid' or 'same'. default to 'same'
    :param split: split channels as used in Alexnet. a int default to 1
    :param w_init_config:
    :param b_init_config: initializer for b. default to zero initializer.
    :param nl: nonlinearity. default to `relu`.
    :param use_bias: whether to use bias. a boolean default to True
    :returns: a NHWC tensor
    """
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride)

    if not w_init_config:
        w_init_config = FillerConfig(type='xavier', variance_norm='AVERAGE')
    if not b_init_config:
        b_init_config = FillerConfig(type='constant', value=0.0)
    w_initializer = get_initializer(filter_shape, w_init_config)
    b_initializer = get_initializer([out_channel], b_init_config)

    W = tf.get_variable('W', filter_shape, initializer=w_initializer)
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_initializer)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding)
    else:
        inputs = tf.split(3, split, x)
        kernels = tf.split(3, split, W)
        outputs = [tf.nn.conv2d(i, k, stride, padding)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(3, outputs)
    return nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')



class BaseConvolution(Layer):
    """"""
    def __init__(self, inputs, params):
        super(BaseConvolution, self).__init__(inputs, params)
        self.weight_creater = WeightCreater()
        self.bias_creater = BisaCreater()

    def _weight_shape(self):
        params = self.params
        channels = self.input_shape[-1]
        num_output = params.num_output
        weight_shape = []
        if params.HasField('kernel_h') and params.HasField('kernel_w'):
            assert not params.HasField('kernel_size')
            weight_shape.append(params.kernel_h)
            weight_shape.append(params.kernel_w)
        else:
            kernel_size_size = len(params.kernel_size)
            assert kernel_size_size == 2
            weight_shape.extend(params.kernel_size)
        weight_shape.append(channels)
        weight_shape.append(num_output)
        return weight_shape

    def _stride_4d(self):
        params = self.params
        strides = [1]
        if params.HasField('stride_h') and params.HasField('stride_w'):
            assert not params.HasField('stride')
            strides.append(params.kernel_h)
            strides.append(params.kernel_w)
        else:
            strides.append(params.stride)
            strides.append(params.stride)
        strides.append(1)
        return strides

    def _weight_process(self):
        self.weight_var = self.weight_creater.create(self.weight_shape, self.params.weight_filler)

    def _bias_process(self):
        self.bias_var = self.bias_creater.create([self.weight_shape[-1]], self.params.bias_filler)

    def _layer_setup(self):
        self.inputs = cast_list(self.inputs)
        assert len(self.inputs) == 1 and isinstance(self.params, pb.ConvolutionParameter)
        # setup input shape
        self.input_shape = tensor_shape(self.inputs[0])
        # setup strids and padding
        self.stride_4d = self._stride_4d()
        self.padding = 'VALID'
        if self.params.pad == pb.ConvolutionParameter.SAME:
            self.padding = 'SAME'
        # setup kernel shape
        self.weight_shape = self._weight_shape()
        # handle weights and bias
        self.bias_term = self.params.bias_term
        self._weight_process()
        if self.bias_term:
            self._bias_process()

    def _forward(self):
        group = self.params.group
        if group == 1:
            self.conv_2d = tf.nn.conv2d(self.inputs[0], self.weight_var, self.stride_4d, self.padding)
        else:
            inputs = tf.split(3, group, self.inputs[0])
            weight_vars = tf.split(3, group, self.weight_var)
            conv_2ds = [tf.nn.conv2d(i, k, self.stride_4d, self.padding)
                       for i, k in zip(inputs, weight_vars)]
            self.conv_2d = tf.concat(3, conv_2ds)
        if self.bias_term:
            self.conv_2d = tf.nn.bias_add(self.conv_2d, self.bias_var)
        self.outputs = [self.conv_2d]

    def _reshape(self):
        pass


    class Conv2D(BaseConvolution):





































