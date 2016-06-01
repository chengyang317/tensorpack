# -*- coding: UTF-8 -*-
# File: paraminit.py
# Author: philipcheng
# Time: 5/31/16 -> 5:46 PM
import tensorflow as tf
import numpy as np

__all__ = ['msra_initializer']


def msra_initializer(channel=None):
    assert not channel
    return tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel))
