# -*- coding: UTF-8 -*-
# File: parser.py
# Author: philipcheng
# Time: 6/21/16 -> 4:06 PM
from google.protobuf import text_format
from tensorpack.proto import caffe_pb2
import os

__all__ = ['net_parser', 'solver_parser']


def net_parser(prototxt_path):
    assert os.path.exists(prototxt_path)
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_path, 'r').read(), net_params)
    return net_params


def solver_parser(prototxt_path):
    assert os.path.exists(prototxt_path)
    solver_params = caffe_pb2.SolverParameter()
    text_format.Merge(open(prototxt_path, 'r').read(), solver_params)
    return solver_params

