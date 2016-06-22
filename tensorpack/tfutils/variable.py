# -*- coding: UTF-8 -*-
# File: variable.py
# Author: philipcheng
# Time: 6/20/16 -> 10:00 PM
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorpack.proto import tensorpack_pb2 as pb
from tensorpack.tfutils.filler import filler_registry

__all__ = ['WeightCreater', 'BisaCreater']


class VariableCreater(object):
    """"""
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.var_name = name
        self.var_shape = None
        self.filler_param = None
        self.var_initializer = None

    def create(self, var_shape, filler_param):
        assert isinstance(var_shape, (tuple, list))
        assert isinstance(filler_param, pb.FillerParameter)
        self.var_shape = var_shape
        self.filler_param = filler_param
        return self._create()

    def get_var(self):
        return tf.get_variable(self.var_name, self.var_shape, initializer=self.var_initializer)

    @abstractmethod
    def _create(self):
        """"""

    def get_initializer(self):
        filler_type_str = self.filler_param.type
        initializer_func = filler_registry.look_up(filler_type_str)
        self.var_initializer = initializer_func(self.var_shape, self.filler_param)
        return self.var_initializer


class WeightCreater(VariableCreater):
    """"""
    def __init__(self, name='weight'):
        super(WeightCreater, self).__init__(name)

    def _create(self):
        initializer = self.get_initializer()
        weight_var = self.get_var()
        return weight_var


class BisaCreater(VariableCreater):
    """"""
    def __init__(self, name='bias'):
        super(BisaCreater, self).__init__(name)

    def _create(self):
        pass








