#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: model_desc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple
from tensorpack.models.net import Net
from tensorpack.utils import logger
from tensorpack.tfutils.common import get_tensor_by_name
from tensorpack.tfutils.gradproc import CheckGradient
from tensorpack.utils.search import get_elem_from_dict

__all__ = ['ModelDesc', 'InputVar']

InputVar = namedtuple('InputVar', ['type', 'shape', 'name'])


class ModelDesc(object):
    """ Base class for a model description """
    __metaclass__ = ABCMeta

    def __init__(self):
        # Collect all the variables which once be created in the build_graph function. Include variables from multi gpu
        # train process and predict process
        self.build_graph_variables = {'train': [], 'predict': []}

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

    def _add_build_graph_variables(self, variables_dict, is_training):
        if is_training:
            self.build_graph_variables['train'].append(variables_dict)
        self.build_graph_variables['predict'].append(variables_dict)

    @abstractmethod
    def _get_input_vars(self):
        """:returns: a list of InputVar """

    def get_tensors_by_names(self, names, is_training=True, name_prefix='', tower=0):
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
        build_graph_variables = self.build_graph_variables['train' if is_training else 'predict'][tower]
        tensors = []
        for name in names:
            tensor = get_elem_from_dict(build_graph_variables, name)
            if tensor is None:
                name = name_prefix + name
                tensor = get_tensor_by_name(name)
            assert tensor is not None, "Can't match any tensor by {}".format(name)
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
        variables_dict = self._build_graph(model_inputs, is_training)
        assert variables_dict is not None, "There has not returned local variables in the _build_graph function"
        self._add_build_graph_variables(variables_dict, is_training)


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


class CaffeModel(ModelDesc):
    """"""
    def __init__(self, prototxt_path):
        super(CaffeModel, self).__init__()
        self.net = Net(prototxt_path)

    def _get_input_vars(self):
        """:returns: a list of InputVar """
        input_tensors = self.net.raw_input_tensors.values()

    def _build_graph(self, model_inputs, is_training):
        self.net.build_net_graph(model_inputs, is_training)





