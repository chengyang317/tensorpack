# -*- coding: UTF-8 -*-
# File: prototxt.py
# Author: philipcheng
# Time: 6/21/16 -> 4:07 PM
from google.protobuf import text_format
from tensorpack.proto import caffe_pb2

Net = caffe_pb2.NetParameter()
text_format.Merge(open("VGG_ILSVRC_16_layers_deploy.prototxt", 'r').read(), Net)
pass
