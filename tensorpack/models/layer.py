# -*- coding: UTF-8 -*-
# File: layer.py
# Author: philipcheng
# Time: 6/20/16 -> 11:11 AM
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorpack.framework.concat import Concat
from tensorpack.framework.registry import Registry
from tensorpack.proto.caffe_pb2 import V1LayerParameter


__all__ = ['Layer', 'layer_class_register']

layer_class_register = Registry('layer_class')
layer_create_func_register = Registry('layer_create_func')


@layer_create_func_register.register('LayerParameter')
def layer_create_from_layerparameter(layer_params):
    layer_class = layer_class_register.look_up(layer_params.type.upper())
    return layer_class(layer_params)


@layer_create_func_register.register('V1LayerParameter')
def layer_create_from_v1layerparameter(layer_params):
    key = V1LayerParameter.LayerType.Name(layer_params.type)
    layer_class = layer_class_register.look_up(key)
    return layer_class(layer_params)


def layer_create(layer_params, layer_params_type):
    layer_create_func = layer_create_func_register.look_up(layer_params_type)
    return layer_create_func(layer_params)


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, inputs, params):
        self.inputs = inputs
        self.params = params

    def layer_setup(self):
        self._layer_setup()

    @abstractmethod
    def _layer_setup(self):
        """"""

    def reshape(self):
        self._reshape()

    @abstractmethod
    def _reshape(self):
        """"""

    def forward(self):
        return self._forward()

    @abstractmethod
    def _forward(self):
        """"""

    def build(self):
        self.layer_setup()
        self.forward()
        self.reshape()


class MultiLayers(Layer, Concat):
    """"""
    def __init__(self, inputs, layer_list, layer_param_list):
        super(MultiLayers, self).__init__(layer_list, layer_param_list)

    def _layer_setup(self):
        pass











