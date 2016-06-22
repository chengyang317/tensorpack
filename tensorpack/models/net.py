# -*- coding: UTF-8 -*-
# File: net.py
# Author: philipcheng
# Time: 6/21/16 -> 4:27 PM
import tensorflow as tf
from tensorpack.proto import net_parser, proto_params_to_dict, caffe_pb2
from tensorpack.models.inputs import inputs_create
from tensorpack.models.layer import layer_create

__all__ = ['Net']


class Net(object):
    """"""
    def __init__(self, net_params=None, net_phase='train'):
        self.net_params = net_params
        self.net_phase = net_phase

    def load_from_prototxt(self, prototxt_path):
        self.net_params = net_parser(prototxt_path)

    def parse_from_params(self):
        assert isinstance(self.net_params, caffe_pb2.NetParameter)
        net_params = proto_params_to_dict(self.net_params)
        if 'name' in net_params:
            self.net_name = net_params['name']
        else:
            self.net_name = 'net'
        if 'input' in net_params:
            self.input_params = net_params['input']
            input_nums = len(self.input_params)
            assert 'input_shape' in net_params or 'input_dim' in net_params
            if 'input_shape' in net_params:
                self.input_shape_params = net_params['input_shape']
                self.input_shape_params_type = 'BlobShape'
            elif 'input_dim' in net_params:
                self.input_shape_params = net_params['input_dim']
                assert len(self.input_shape_params) == 4 * input_nums
                self.input_shape_params_type = 'dim'
            else:
                raise BaseException("net params must have input and input_shape or input_dim meanwhile")
        if 'state' in net_params:
            self.net_state_params = net_params['state']
        if 'lay' in net_params:
            self.lays_params = net_params['lay']
            self.lays_params_type = 'LayerParameter'
        elif 'lays' in net_params:
            self.lays_params = net_params['lays']
            self.lays_params_type = 'V1LayerParameter'
        else:
            raise BaseException("net parms does not contain layer params")

    def build_inputs(self):
        return inputs_create(self.input_params, self.input_shape_params, self.input_shape_params_type)

    def build_layers(self):
        ret_layers = []
        for layer_params in self.lays_params:
            ret_layers.append(layer_create(layer_params, self.lays_params_type))
        return ret_layers

    def filt_layers(self):
        phase_map = {0: 'train', 1: 'test'}
        filtered_layer_names = set()
        filtered_layers = []
        for layer_params in self.lays_params:
            phase = self.phase
            if len(layer.include):
                phase = phase_map[layer.include[0].phase]
            if len(layer.exclude):
                phase = phase_map[1 - layer.include[0].phase]
            exclude = (phase != self.phase)
            # Dropout layers appear in a fair number of Caffe
            # test-time networks. These are just ignored. We'll
            # filter them out here.
            if (not exclude) and (phase == 'test'):
                exclude = (layer.type == LayerType.Dropout)
            if not exclude:
                filtered_layers.append(layer)
                # Guard against dupes.
                assert layer.name not in filtered_layer_names
                filtered_layer_names.add(layer.name)
        return filtered_layers


    def build_graph(self):
        if hasattr('input_params', Net):
            self.inputs = self.build_inputs()

















