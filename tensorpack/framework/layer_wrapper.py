# -*- coding: UTF-8 -*-
# File: layer_wrapper
# Author: Philip Cheng
# Time: 6/23/16 -> 12:32 AM
from tensorpack.models.layer import Layer

class FuncToLayer(object):
    def __init__(self):
        pass

    def convert(self, layer_func):
        layer = Layer('tf')
        def wrap_func(*args, **kwargs):

            pass
        return wrap_func




