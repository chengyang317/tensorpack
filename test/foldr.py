# -*- coding: UTF-8 -*-
# File: foldr.py
# Author: philipcheng
# Time: 6/17/16 -> 9:41 AM
import tensorflow as tf
import numpy as np
from tensorpack import create_shape

def func(a,b):
    a_shape = create_shape(a)
    b_shape = create_shape(b)
    print(a_shape)
    print(b_shape)
    # print(len(a_shape))
    # print(len(b_shape))
    # if len(a_shape) == 1:
    #     a = tf.expand_dims(a, 0)
    # if len(b_shape) == 1:
    #     b = tf.expand_dims(b, 0)
    # a_shape = get_shape(a)
    # b_shape = get_shape(b)
    # print(a_shape)
    # print(b_shape)
    # print(len(a_shape))
    # print(len(b_shape))
    # if len(a_shape) >= 3:
    #     a = tf.squeeze(a)
    # if len(b_shape) >= 3:
    #     b = tf.squeeze(b)

    # if len(a.get_shape().as_list()) == 1:
    #     a = tf.expand_dims(a, dim=0)
    # if len(b.get_shape().as_list()) == 1:
    #     b = tf.expand_dims(b, dim=0)
    a = tf.reshape(a, (-1,2,2))
    b = tf.reshape(b, (-1,2,2))
    # a = a+5
    b = b + 5
    c = tf.concat(0, [a, b])
    c_shape = create_shape(c)
    print(c_shape)
    return c


x = tf.placeholder(tf.float32, shape=[None, 2, 2])
# sum = tf.foldl(lambda a, b: b, x, initializer=1.0)
sum = tf.foldl(func, x, initializer=None)
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