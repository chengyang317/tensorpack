# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: philipcheng
# Time: 6/8/16 -> 9:42 AM
import tensorflow as tf
import argparse
import tensorpack as tp
import os
import numpy as np

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

BATCH_SIZE = 128
IMAGE_SIZE = 28


def local_net(is_training):
    def func(x):
        l = tp.FullyConnected('fc_local0', x, num_output=20, nl=tf.nn.tanh)
        if is_training:
            l = tf.nn.dropout(l, 0.5)
        l = tp.FullyConnected('fc_local1', l, num_output=6, nl=tf.nn.tanh,
                              bias_filler=tp.FillerConfig(type='custom', value=np.array([1., 0, 0, 0, 1., 0])))
        return l
    return func


class Model(tp.ModelDesc):
    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                tp.InputVar(tf.int32, (None,), 'label')
               ]

    def _build_graph(self, input_vars, is_training):
        is_training = bool(is_training)
        keep_prob = tf.constant(0.5 if is_training else 1.0)
        image, label = input_vars
        image = tf.expand_dims(image, 3)    # add a single channel
        nl = tp.PReLU.f
        image = image * 2 - 1
        # add spatial_transformer layer
        image = tp.spatial_transformer('spatial', image, local_net=local_net(is_training), out_shape=(40, 40))
        l = tp.Conv2D('conv0', image, num_output=32, kernel_size=3, nl=nl, pad='VALID')
        l = tp.MaxPooling('pool0', l, 2)
        l = tp.Conv2D('conv1', l, num_output=32, kernel_size=3, nl=nl, pad='SAME')
        l = tp.Conv2D('conv2', l, num_output=32, kernel_size=3, nl=nl, pad='VALID')
        l = tp.MaxPooling('pool1', l, 2)
        l = tp.Conv2D('conv3', l, num_output=32, kernel_size=3, nl=nl, pad='VALID')

        l = tp.FullyConnected('fc0', l, 512)
        l = tf.nn.dropout(l, keep_prob)

        # fc will have activation summary by default. disable this for the output layer
        logits = tp.FullyConnected('fc1', l, num_output=10, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        tf.add_to_collection(tp.MOVING_SUMMARY_VARS_KEY, cost)

        # compute the number of failed samples, for ClassificationError to use at test time
        wrong = tp.prediction_incorrect(logits, label)
        nr_wrong = tf.reduce_sum(wrong, name='wrong')
        # monitor training error
        tf.add_to_collection(
            tp.MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_cost = tf.mul(1e-5,
                         tp.regularize_cost('fc.*/W', tf.nn.l2_loss),
                         name='regularize_loss')
        tf.add_to_collection(tp.MOVING_SUMMARY_VARS_KEY, wd_cost)

        tp.add_param_summary([('.*/W', ['histogram'])])   # monitor histogram of all W
        self.cost = tf.add_n([wd_cost, cost], name='cost')


def get_config():
    basename = os.path.basename(__file__)
    tp.logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]), 'd')

    # prepare dataset
    dataset_train = tp.BatchData(tp.dataset.Mnist('train'), 128)
    dataset_test = tp.BatchData(tp.dataset.Mnist('test'), 256, remainder=True)
    step_per_epoch = dataset_train.size()

    lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=tp.get_global_step_var(),
        decay_steps=dataset_train.size() * 10,
        decay_rate=0.3, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return tp.TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=tp.Callbacks([
            tp.StatPrinter(),
            tp.ModelSaver(),
            tp.InferenceRunner(dataset_test,
                [tp.ScalarStats('cost'), tp.ClassificationError()])
        ]),
        session_config=tp.get_default_sess_config(0.5),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
        nr_tower=1
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.'
                        ) # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = tp.SaverRestore(args.load)
        tp.QueueInputTrainer(config).train()

