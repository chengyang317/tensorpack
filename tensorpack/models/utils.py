# -*- coding: UTF-8 -*-
# File: utils.py
# Author: philipcheng
# Time: 6/18/16 -> 2:14 AM
from functools import wraps
import six
import copy
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.modelutils import *
from tensorpack.tfutils.summary import add_activation_summary
import tensorflow as tf

__all__ = ['layer', 'shape2d', 'shape4d']


class Layer(object):
    """
    Layes
    """
    def __init__(self):
        self.layer_logged = set()
        self.layer_outputs = {}
        self.layer_names = []
        self.layer_funcs = []
        self.layer_names_funcs = {}

    def register(self, summary_activation=False, log_shape=True):
        def wrapper(layer_func):
            @wraps(layer_func)
            def wrapped_func(*args, **kwargs):
                layer_name = args[0]
                assert isinstance(layer_name, six.string_types), \
                    'name must be the first argument. Args: {}'.format(args)
                args = args[1:]
                do_summary = kwargs.pop('summary_activation', summary_activation)
                inputs = args[0]
                # update from current argument scope
                actual_args = copy.copy(argscope.get_arg_scope()[layer_func.__name__])
                actual_args.update(kwargs)

                with tf.variable_scope(layer_name) as scope:
                    do_log_shape = log_shape and scope.name not in self.layer_logged
                    do_summary = do_summary and scope.name not in self.layer_logged
                    if do_log_shape:
                        logger.info("{} input: {}".format(scope.name, get_shape_str(inputs)))

                    # run the actual layer function
                    outputs = layer_func(*args, **actual_args)
                    self.layer_outputs[scope.name] = outputs
                    self.layer_names.append(scope.name)
                    self.layer_funcs.append(layer_func)
                    self.layer_names_funcs[scope.name] = layer_func

                    if do_log_shape:
                        # log shape info and add activation
                        logger.info("{} output: {}".format(
                            scope.name, get_shape_str(outputs)))
                        self.layer_logged.add(scope.name)
                    if do_summary:
                        if isinstance(outputs, list):
                            for x in outputs:
                                add_activation_summary(x, scope.name)
                        else:
                            add_activation_summary(outputs, scope.name)
                    return outputs
            wrapped_func.f = layer_func  # attribute to access the underlining function object
            return wrapped_func
        return wrapper


layer = Layer()


def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a):
    # for use with tensorflow
    return [1] + shape2d(a) + [1]

