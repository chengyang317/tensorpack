# -*- coding: UTF-8 -*-
# File: layer.py
# Author: philipcheng
# Time: 6/20/16 -> 11:11 AM
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorpack.framework.concat import Concat
from tensorpack.framework.registry import Registry
from tensorpack.proto.caffe_pb2 import V1LayerParameter, LayerParameter
from tensorpack.proto import lay_params_parser


__all__ = ['Layer', 'layer_class_register', 'layer_create', 'layers_create']

layer_class_register = Registry('layer_class')


def layer_create(layer_params):
    layer_type = layer_params.type
    if isinstance(layer_type, int):
        layer_type = V1LayerParameter.LayerType.Name(layer_type)
    else:
        layer_type = layer_type.upper()
    layer_class = layer_class_register.look_up(layer_type)
    return layer_class(lay_params=layer_params)


def layers_create(net_tensors, net_params):
    ret_layers = []
    if net_params.HasField('layer'):
        layers_params = net_params.layer
    else:
        layers_params = net_params.layers
    for layer_params in layers_params:
        ret_layers.append(layer_create('caffe', net_tensors, layer_params))
    return ret_layers


class Layer(object):
    __metaclass__ = ABCMeta

    def __new__(cls, *args, **kwargs):
        obj = super(Layer, cls).__new__(cls)
        if args[0] == 'caffe':
            cls.__init__ = cls._caffe_init
        else:
            cls.__init__ = cls._tf_init
        return obj

    def _tf_init(self, *args, **kwargs):
        self.mode = 'tf'
        self.forward_params = (args, kwargs)
        self.build_layer_graph()

    def _caffe_init(self, layer_params):
        self.mode = 'caffe'
        assert isinstance(layer_params, (V1LayerParameter, LayerParameter))
        self.net_tensors = {}
        self.lay_params = layer_params

    def layer_setup(self):
        """
        Only in caffe mode, it will be called
        :return:
        """
        assert all(bottom in self.net_tensors for bottom in self.lay_params.bottom)
        self.input_tensors = {bottom: self.net_tensors[bottom] for bottom in self.lay_params.bottom}
        forward_kwargs = self._layer_setup(self.input_tensors, self.lay_params)
        self.forward_params = ([], forward_kwargs)

    @abstractmethod
    def _layer_setup(self, input_tensors, lay_params):
        """
        :param input_tensors:
        :param lay_params:
        :return: forward_kwargs
        """

    def forward(self):
        forward_params = self.forward_params
        return self._forward(*forward_params[0], **forward_params[1])

    @abstractmethod
    def _forward(self):
        """"""

    def layer_func(self, input_tensors, **kwargs):
        pass

    def build_layer_graph(self, net_tensors=None):
        if self.mode == 'caffe':
            self.layer_setup()
            self.net_tensors.update(net_tensors)
        outputs = self.forward()
        assert len(outputs) == len(self.lay_params.top)
        self.output_tensors = {top: outputs[ind] for ind, top in enumerate(self.lay_params.top)}











