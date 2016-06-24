# -*- coding: UTF-8 -*-
# File: parser.py
# Author: philipcheng
# Time: 6/21/16 -> 4:06 PM
import os
from google.protobuf import text_format
from tensorpack.proto import caffe_pb2
from tensorpack.proto.utils import filter_proto_params
from tensorpack.framework.registry import Registry

__all__ = ['txt_to_net_params', 'txt_to_solver_params', 'net_params_parser', 'lay_params_parser']





def net_params_parser(net_params):
    ret_dict = {}
    net_params = filter_proto_params(net_params)
    if 'name' in net_params:
        ret_dict['name_params'] = net_params['name']

    if 'input' in net_params:
        input_params = net_params['input']
        input_nums = len(input_params)
        assert 'input_shape' in net_params or 'input_dim' in net_params
        if 'input_shape' in net_params:
            input_shape_params = net_params['input_shape']
            input_shape_params_type = 'BlobShape'
        elif 'input_dim' in net_params:
            input_shape_params = net_params['input_dim']
            assert len(input_shape_params) == 4 * input_nums
            input_shape_params_type = 'dim'
        else:
            raise BaseException("net params must have input and input_shape or input_dim meanwhile")
        ret_dict['input_shape_params'] = input_shape_params
        ret_dict['input_shape_params_type'] = input_shape_params_type

    if 'state' in net_params:
        ret_dict['state_params'] = net_params['state']

    if 'lay' in net_params:
        lays_params = net_params['lay']
        lays_params_type = 'LayerParameter'
    elif 'lays' in net_params:
        lays_params = net_params['lays']
        lays_params_type = 'V1LayerParameter'
    else:
        raise BaseException("net parms does not contain layer params")
    ret_dict['lays_params'] = lays_params
    ret_dict['lays_params_type'] = lays_params_type
    return ret_dict


def lay_params_parser(lay_params):
    ret_dict = {}

    if 'name' in lay_params:
        ret_dict['name_params'] = lay_params['name']

    if 'bottom' in lay_params:
        ret_dict['bottom_params'] = lay_params['bottom']
    if 'top' in lay_params:
        ret_dict['top_params'] = lay_params['top']

    lay_type_params = []
    for params_name in lay_params.keys():
        if params_name.endswith('_param'):
            lay_type_params.append(lay_params[params_name])
    ret_dict['lay_type_params'] = lay_type_params
    return ret_dict


lay_type_params_parser_reigster = Registry('lay_type_params_parser_reigster')


@lay_type_params_parser_reigster.register('InnerProductParameter'.upper())
def inner_product_parameter(params):
    ret_dict = {}
    required_fields = ('num_output', 'weight_filler')
    assert all(key in params for key in required_fields)
    ret_dict.update({key: params[key] for key in required_fields})
    optional_fields = ('bias_filler',)
    ret_dict.update({key: params[key] for key in required_fields if key in params})
    return ret_dict































