# -*- coding: UTF-8 -*-
# File: concat.py
# Author: philipcheng
# Time: 6/21/16 -> 11:22 AM
from abc import ABCMeta, abstractmethod

__all__ = ['Concat']


class Concat(object):
    __metaclass__ = ABCMeta
    """"""
    def __init__(self, first_input, object_list, object_param_list):
        self.first_input = first_input
        assert len(object_list) == len(object_param_list)
        self.object_list = object_list
        self.object_param_list = self.object_param_list

    def iter_call(self, attr):
        for obj in self.object_list:
            if not hasattr(obj, attr):
                raise LookupError("{} does not have attribute {}".format(obj.__name__), str(attr))
            if not hasattr(obj.attr, '__call__'):
                raise LookupError("{} is not callable".format(obj.attr.__name__))
        input = self.first_input
        for obj, obj_param in zip(self.object_list, self.object_param_list):
            input = obj.attr(input, obj_param)










