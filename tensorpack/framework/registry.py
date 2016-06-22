# -*- coding: UTF-8 -*-
# File: registry.py
# Author: philipcheng
# Time: 6/20/16 -> 10:34 PM
from functools import wraps

__all__ = ['Registry']


class Registry(object):
    """"""
    def __init__(self, register_name):
        self._name = register_name
        self.registry = {}

    def register(self, key=None):
        """"""
        def wrapper(object):
            if key is None:
                key_local = object.__name__
                if key_local in self.registry:
                    raise KeyError("{} once be registered in registry {}".format(key, self._name))
                self.registry[key_local] = object
            else:
                self.registry[key] = object
            return object
        return wrapper

    def look_up(self, key):
        if key in self.registry:
            return self.registry[key]
        raise LookupError("{} registry has no entry for: {}".format(self._name, key))






