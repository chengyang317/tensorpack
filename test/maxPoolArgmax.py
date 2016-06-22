# -*- coding: UTF-8 -*-
# File: maxPoolArgmax.py
# Author: philipcheng
# Time: 6/2/16 -> 3:20 PM
import numpy as np
import tensorflow as tf
import tensorpack as tp


if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:1'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        i = np.array(range(64))
        i = tf.constant(i, shape=(2,4,4,2), dtype=tf.float32)
        out, index = tf.nn.max_pool_with_argmax(i, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        unpool = tp.ArgmaxUnPooling('unpooling', out, index, shape_scale=2)
        i_v, out_v, index_v, unpool_v = sess.run([i, out, index, unpool])
        # i_v, out_v, index_v = sess.run([i, out, index])
        pass


