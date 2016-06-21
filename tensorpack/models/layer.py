# -*- coding: UTF-8 -*-
# File: layer.py
# Author: philipcheng
# Time: 6/20/16 -> 11:11 AM
from abc import ABCMeta, abstractmethod
import tensorflow as tf


__all__ = ['Layer']


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, inputs, params):
        self.inputs = inputs
        self.params = params

    def layer_setup(self):
        self._layer_setup()

    @abstractmethod
    def _layer_setup(self):
        """"""

    def reshape(self):
        self._reshape()

    @abstractmethod
    def _reshape(self):
        """"""

    def forward(self):
        return self._forward()

    @abstractmethod
    def _forward(self):
        """"""

    def build(self):
        self.layer_setup()
        self.forward()
        self.reshape()









