# -*- coding: UTF-8 -*-
# File: map_fn.py
# Author: philipcheng
# Time: 6/17/16 -> 3:04 PM
import tensorflow as tf
import numpy as np
from tensorpack import create_shape


def func(a):
    a_shape = create_shape(a)
    # a = tf.reshape(a, (-1,2,2))
    print(a_shape)
    a = a + 5
    return a


x = tf.placeholder(tf.float32, shape=[None, 2, 2])
# sum = tf.foldl(lambda a, b: b, x, initializer=1.0)
sum = tf.map_fn(func, x)
with tf.Session() as sess:
  a = np.array(range(16))
  a = a.reshape((4,2,2))
  # for i in range(100):
  #   length = np.random.randint(0,10)
  #   a = np.random.randint(0, 10, length)
  # print sess.run(sum,feed_dict={x:a})
  sum1 = tf.reshape(sum, [-1])
  sum_eval = sess.run(sum, feed_dict={x: a})
  print(sum_eval)