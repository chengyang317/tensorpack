# -*- coding: UTF-8 -*-
# File: dropout.py
# Author: philipcheng
# Time: 6/18/16 -> 9:41 PM
import tensorflow as tf
from tensorpack.models.utils import layer

__all__ = ['dropout']


@layer.register()
def dropout(x, prob):
    return tf.nn.dropout(x, prob, name='output')