#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple
import re
from tensorpack.utils import logger
from tensorpack.tfutils.common import get_tensor_by_name

__all__ = ['ModelDesc', 'InputVar']

InputVar = namedtuple('InputVar', ['type', 'shape', 'name'])


class ModelDesc(object):
    """ Base class for a model description """
    __metaclass__ = ABCMeta

    def get_input_vars(self):
        """
        Create or return (if already created) raw input TF placeholder vars in the graph.

        :returns: the list of raw input vars in the graph
        """
        try:
            return self.reuse_input_vars()
        except KeyError:
            pass
        input_vars = self._get_input_vars()
        ret = []
        for v in input_vars:
            ret.append(tf.placeholder(v.type, shape=v.shape, name=v.name))
        return ret

    def reuse_input_vars(self):
        """ Find and return already-defined input_vars in default graph"""
        input_var_names = [k.name for k in self._get_input_vars()]
        g = tf.get_default_graph()
        return [g.get_tensor_by_name(name + ":0") for name in input_var_names]

    @abstractmethod
    def _get_input_vars(self):
        """:returns: a list of InputVar """

    def _get_tensor_from_build(self, name, tower=0):
        """
        Find matched tensor in the variables dict created in _build_graph process
        :param name: maybe a full name of a variable or a regex
        :param tower: gpu index
        :return: matched tensor or error
        """
        variables_dict = self.build_graph_variables[tower]
        tensor = variables_dict.get(name, None)
        if tensor is None:
            keys = variables_dict.keys
            matched_keys = []
            for key in keys:
                if re.search(name, key):
                    matched_keys.append(key)
            assert len(matched_keys) <= 1, "{} matched multi variabls {}".format(name, ','.join(matched_keys))
            if len(matched_keys) == 0:
                return None
            tensor = variables_dict[matched_keys[0]]
        return tensor

    def get_tensors_by_names(self, names, tower=0):
        """
        Find matched tensors by names. The match process is firstly to match variables created in _build_graph func.
        And the to match target in the whole tensorflow graph.
        :param names: name or a list of names. name maybe is in regex form or a specified name defined in the graph
         form or in python locals vairable name form
        :param tower: gpu index
        :return: matched tensors which have the length as names' length
        """
        if not isinstance(names, (tuple, list)):
            names = list(names)
        tensors = []
        for name in names:
            tensor = self._get_tensor_from_build(name, tower)
            if tensor is None:
                name = 'tower{}/'.format(tower) + name
                tensor = get_tensor_by_name(name)
            assert tensor is None, "Can't match any tensor by {}".format(name)
            tensors.append(tensor)
        return tensors

    def build_graph(self, model_inputs, is_training):
        """
        setup the whole graph.
        :param model_inputs: a list of input variable in the graph
            e.g.: [image_var, label_var] with:

            * image_var: bx28x28
            * label_var: bx1 integer
        :param is_training: a boolean
        :returns: the cost to minimize. a scalar variable
        """
        self.build_graph_variables = [] #used to store multi graph elemtents in multigpu case, index is gpu index
        self._build_graph(model_inputs, is_training)
        assert not self.build_graph_variables, \
            "During _build_graph func, there has not added local variables in to build_graph_variables"

    #@abstractmethod
    def _build_graph(self, inputs, is_training):
        if self._old_version():
            self.model_inputs = inputs
            self.is_training = is_training
        else:
            raise NotImplementedError()

    def _old_version(self):
        # for backward-compat only.
        import inspect
        args = inspect.getargspec(self._get_loss)
        return len(args.args) == 3

    def get_loss(self):
        if self._old_version():
            assert type(self.is_training) == bool
            logger.warn("!!!using _get_cost to setup the graph is deprecated in favor of _build_graph")
            logger.warn("See examples for details.")
            return self._get_loss(self.model_inputs, self.is_training)
        else:
            return self._get_loss()

    def _get_loss(self, *args):
        return self.loss

    def get_gradient_processor(self):
        """ Return a list of GradientProcessor. They will be executed in order"""
        return [CheckGradient()]#, SummaryGradient()]

