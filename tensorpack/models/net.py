# -*- coding: UTF-8 -*-
# File: net.py
# Author: philipcheng
# Time: 6/21/16 -> 4:27 PM
from tensorpack.proto.utils import filter_proto_by_regex, filter_layers_params, txt_to_net_params
from tensorpack.models.inputs import inputs_create_from_input
from tensorpack.models.layer import layer_create, layers_create

__all__ = ['Net']


class Net(object):
    """"""
    def __init__(self, prototxt_path=None, net_params=None, net_name='net'):
        if prototxt_path:
            assert net_params is None
            self.raw_net_params = txt_to_net_params(prototxt_path)
        self.net_name = net_name
        self.net_tensors = {}
        self.layers = None
        self.net_params = None
        self.raw_input_tensors = self._try_build_raw_input_tensors()

    def _net_params_parser(self, raw_net_params, is_training):
        raw_net_params.name = raw_net_params.name or self.net_name
        raw_net_params.state.phase = 0 if is_training else 1
        filter_layers_params(raw_net_params)
        return raw_net_params

    def create_input_tensors(self, model_inputs):
        assert len(model_inputs) == len(self.raw_input_tensors)
        model_input_tensors = {key: model_inputs[ind] for ind, key in self.raw_input_tensors.keys()}
        return model_input_tensors

    def _try_build_raw_input_tensors(self):
        if self.net_params.HasField('input'):
            return inputs_create_from_input(self.net_params)
        data_layer_params = filter_proto_by_regex(self.net_params, '.*data_param').values()
        trans

        raise EnvironmentError("prototxt has not defined raw inputs")


    def build_layers_graph(self):
        net_tensors = self.net_tensors
        for layer in self.layers:
            layer.build_layer_graph(net_tensors)
            net_tensors.update(layer.output_tensors)

    def build_net_graph(self, model_inputs, is_training):
        self.net_tensors.update(self.create_input_tensors(model_inputs))
        self.net_params = self._net_params_parser(self.raw_net_params, is_training)
        self.layers = layers_create(self.net_params)
        self.build_layers_graph()

















