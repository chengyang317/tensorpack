#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argscope.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from contextlib import contextmanager
from collections import defaultdict
import inspect
import copy
import six

__all__ = ['argscope']


class Argscope(object):
    def __init__(self):
        self.argscope_stack = []

    @staticmethod
    def check_args_exist(layer_func, params):
        layer_func_args = inspect.getargspec(layer_func).args
        for k, v in six.iteritems(params):
            assert k in layer_func_args, "No argument {} in {}".format(k, layer_func.__name__)

    def get_arg_scope(self):
        if len(self.argscope_stack) > 0:
            return self.argscope_stack[-1]
        else:
            return defaultdict(dict)

    @contextmanager
    def scope(self, layer_funcs, **kwargs):
        """

        :param layer_funcs: such as Conv2D
        :param kwargs:
        :return:
        """
        params = kwargs
        if not isinstance(layer_funcs, list):
            layer_funcs = [layer_funcs]
        for layer_func in layer_funcs:
            assert hasattr(layer_func, 'f'), "{} is not a registered layer".format(layer_func.__name__)
            self.check_args_exist(layer_func.f, params)
        new_scope = copy.copy(self.get_arg_scope())
        for layer_func in layer_funcs:
            new_scope[layer_func.__name__].update(params)
        self.argscope_stack.append(new_scope)
        yield
        del self.argscope_stack[-1]


argscope = Argscope()
