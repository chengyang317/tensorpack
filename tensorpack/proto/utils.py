# -*- coding: UTF-8 -*-
# File: utils.py
# Author: philipcheng
# Time: 6/21/16 -> 4:46 PM
import re
import os
from tensorpack.proto import caffe_pb2
from google.protobuf import text_format

__all__ = ['filter_proto_by_regex', 'filter_proto_params', 'check_layer_active', 'filter_layers_params',
           'txt_to_net_params', 'txt_to_solver_params']


def txt_to_net_params(prototxt_path):
    assert os.path.exists(prototxt_path)
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_path, 'r').read(), net_params)
    return net_params


def txt_to_solver_params(prototxt_path):
    assert os.path.exists(prototxt_path)
    solver_params = caffe_pb2.SolverParameter()
    text_format.Merge(open(prototxt_path, 'r').read(), solver_params)
    return solver_params


def filter_proto_by_regex(proto_params, regex):
    ret_dict = dict()
    list_fields_dict = filter_proto_params(proto_params)
    for field_name in list_fields_dict.keys():
        if re.search(regex, field_name):
            ret_dict[field_name] = list_fields_dict[field_name]
    return ret_dict


def filter_proto_params(proto_params):
    list_fields = proto_params.ListFields()
    ret_dict = dict()
    for field_descriptor, value in list_fields:
        ret_dict[field_descriptor.name] = value
    return ret_dict


def check_layer_active(state_params, state_rule_params):
    if state_params.HasField('phase') and state_rule_params.HasField('phase'):
        if state_params.phase != state_rule_params.phase:
            return False
    if state_params.HasField('level'):
        if state_rule_params.HasField('min_level'):
            if state_params.level < state_rule_params.min_level:
                return False
        if state_rule_params.HasField('max_level'):
            if state_params.level > state_rule_params.max_level:
                return False
    if state_params.HasField('stage'):
        state_stage_set = set(state_params.stage)
        for state_rule_stage in state_rule_params.stage:
            state_rule_stage_set = set(state_rule_stage)
            if not state_stage_set.issuperset(state_rule_stage_set):
                return False
        for state_rule_notstage in state_rule_params.not_stage:
            state_rule_notstage_set = set(state_rule_notstage)
            if state_stage_set.issuperset(state_rule_notstage_set):
                return False
    return True


def filter_layers_params(net_params):
    if net_params.HasField('layer'):
        lays_params = net_params.layer
    elif net_params.HasField('layers'):
        lays_params = net_params.layers
    else:
        raise EnvironmentError("prototxt file has not define layers")
    for lay_params in lays_params:
        for include_params in lay_params.include:
            if not check_layer_active(net_params.state, include_params):
                lays_params.pop(lay_params)
        for exclude_params in lay_params.exclude:
            if check_layer_active(net_params.state, exclude_params):
                lays_params.pop(lay_params)





































