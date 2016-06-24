#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dropout.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
from contextlib import contextmanager
from copy import copy
import re
import six
import tensorflow as tf
from tensorpack.utils.naming import *
from tensorpack.tfutils.modelutils import get_name_str

__all__ = ['get_default_sess_config', 'get_global_step', 'get_global_step_var', 'get_op_var_name', 'get_vars_by_names',
           'get_elems_by_keys', 'backup_collection', 'restore_collection', 'clear_collection', 'freeze_collection',
           'get_name_str', 'get_op_by_name', 'get_tensor_by_name']


def get_default_sess_config(mem_fraction=0.9):
    """
    Return a better session config to use as default.
    Tensorflow default session config consume too much resources.

    :param mem_fraction: fraction of memory to use.
    :returns: a `tf.ConfigProto` object.
    """
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    #conf.log_device_placement = True
    return conf


def get_global_step_var():
    """ :returns: the global_step variable in the current graph. create if not existed"""
    try:
        return tf.get_default_graph().get_tensor_by_name(GLOBAL_STEP_VAR_NAME)
    except KeyError:
        var = tf.Variable(
            0, trainable=False, name=GLOBAL_STEP_OP_NAME)
        return var


def get_global_step():
    """ :returns: global_step value in current graph and session"""
    return tf.train.global_step(
        tf.get_default_session(),
        get_global_step_var())


def get_op_var_name(name):
    """
    Variable name is assumed to be ``op_name + ':0'``

    :param name: an op or a variable name
    :returns: (op_name, variable_name)
    """
    if name[-2] == ':':
        return name[:-2], name
    else:
        return name, name + ':0'


def get_vars_by_names(names):
    """
    Get a list of variables in the default graph by a list of names
    """
    ret = []
    G = tf.get_default_graph()
    for n in names:
        opn, varn = get_op_var_name(n)
        ret.append(G.get_tensor_by_name(varn))
    return ret


def get_elems_by_keys(elements, names):
    """
    Get a list of elements from graph env by locals's key name
    :param elements: A dict contains elements created in the graph builting process.
    :param names: keys of dict
    :return:
    """
    ret = []
    if type(names) not in [tuple, list]:
        names = [names,]
    for key in names:
        try:
            elem = elements[key]
        except:
            raise KeyError('loclas dont have the key {}'.format(key))
        if type(elem) in [tuple, list]:
            ret.extend(elem)
        else:
            ret.append(elem)
    return ret


def backup_collection(keys):
    ret = {}
    for k in keys:
        ret[k] = copy(tf.get_collection(k))
    return ret


def restore_collection(backup):
    for k, v in six.iteritems(backup):
        del tf.get_collection_ref(k)[:]
        tf.get_collection_ref(k).extend(v)


def clear_collection(keys):
    for k in keys:
        del tf.get_collection_ref(k)[:]


@contextmanager
def freeze_collection(keys):
    backup = backup_collection(keys)
    yield
    restore_collection(backup)


def get_op_by_name(name):
    """"""
    G = tf.get_default_graph()
    try:
        op = G.get_operation_by_name(name)
    except:
        op = None
    if op is None:
        g_ops = G.get_operations()
        ops = []
        for g_op in g_ops:
            g_op_name = g_op.name
            if re.search(name, g_op_name):
                ops.append(g_op)
        assert len(ops) <= 1, "{} matched ops: {}".format(name, get_name_str(ops))
        if len(ops) == 0:
            return None
        op = ops[0]
    return op


def get_tensor_by_name(name):
    """"""
    op_name, tensor_name = get_op_var_name(name)
    G = tf.get_default_graph()
    try:
        tensor = G.get_tensor_by_name(tensor_name)
    except:
        tensor = None
    if tensor is None:
        op = get_op_by_name(op_name)
        if op is None:
            return None
        tensor = op.outputs[0]
    return tensor


