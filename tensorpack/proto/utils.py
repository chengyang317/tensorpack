# -*- coding: UTF-8 -*-
# File: utils.py
# Author: philipcheng
# Time: 6/21/16 -> 4:46 PM

__all__ = ['proto_params_to_dict']


def proto_params_to_dict(proto_params):
    list_fields = proto_params.ListFields()
    ret_dict = dict()
    for field_descriptor, value in list_fields:
        ret_dict[field_descriptor.name] = value
    return ret_dict

def extract_condition_from_state(state_params):
    if 'phase'

def check_layer_active(net_state_params, net_state_rule_params):
    active = None
    if net_state_params.phase == net_state_rule_params.phase:
        active = True
        if net_state_params.HasField('level')

