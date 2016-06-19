#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar-convnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import numpy
import tensorflow as tf
import argparse
import numpy as np
import os
import tensorpack as tp

"""
A small convnet model for cifar 10 or cifar100 dataset.

For Cifar10: 90% validation accuracy after 40k step.
"""

class Model(tp.ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, [None, 30, 30, 3], 'input'),
                tp.InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars, is_training):
        image, labels = input_vars
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.image_summary("train_image", image, 10)

        image = image / 4.0     # just to make range smaller
        with tp.argscope.scope(tp.Conv2D, nl=tp.BNReLU(is_training), use_bias=False, kernel_shape=3):
            l = tp.Conv2D('conv1.1', image, out_channel=64)
            l = tp.Conv2D('conv1.2', l, out_channel=64)
            l = tp.MaxPooling('pool1', l, 3, stride=2, padding='SAME')

            l = tp.Conv2D('conv2.1', l, out_channel=128)
            l = tp.Conv2D('conv2.2', l, out_channel=128)
            l = tp.MaxPooling('pool2', l, 3, stride=2, padding='SAME')

            l = tp.Conv2D('conv3.1', l, out_channel=128, padding='VALID')
            l = tp.Conv2D('conv3.2', l, out_channel=128, padding='VALID')
        l = tp.FullyConnected('fc0', l, 1024 + 512, b_init_config=tp.FillerConfig(type='constant', value=0.1))
        l = tp.dropout('fc0_dropout', l, keep_prob)
        l = tp.FullyConnected('fc1', l, 512, b_init_config=tp.FillerConfig(type='constant', value=0.1))
        logits = tp.FullyConnected('linear', l, out_dim=self.cifar_classnum, nl=tf.identity)

        classfication_loss = tp.classification_loss('classfication_loss', logits, labels,
                                                    keys=tp.MOVING_SUMMARY_VARS_KEY)
        nr_wrong = tp.classification_accuracy('nr_wrong', logits, labels, keys=tp.MOVING_SUMMARY_VARS_KEY)
        wd_loss = tp.regularize_loss('wd_loss', 'fc.*/W', 0.004, keys=tp.MOVING_SUMMARY_VARS_KEY)

        tp.add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.loss = tp.sum_loss('sum_loss', [classfication_loss, wd_loss])

def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = tp.dataset.Cifar10(train_or_test)
    else:
        ds = tp.dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            tp.imgaug.RandomCrop((30, 30)),
            tp.imgaug.Flip(horiz=True),
            tp.imgaug.Brightness(63),
            tp.imgaug.Contrast((0.2,1.8)),
            tp.imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (30,30), 0.2, 3),
            tp.imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            tp.imgaug.CenterCrop((30, 30)),
            tp.imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = tp.AugmentImageComponent(ds, augmentors)
    ds = tp.BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = tp.PrefetchDataZMQ(ds, 5)
    return ds

def get_config(cifar_classnum):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    # step_per_epoch = dataset_train.size()
    step_per_epoch = 5
    dataset_test = get_data('test', cifar_classnum)

    sess_config = tp.get_default_sess_config(0.5)

    nr_gpu = tp.get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=tp.get_global_step_var(),
        decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
        decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return tp.TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=tp.Callbacks([
            tp.StatPrinter(),
            tp.ModelSaver(),
            tp.InferenceRunner(dataset_test, tp.ClassificationError())
        ]),
        session_config=sess_config,
        model=Model(cifar_classnum),
        step_per_epoch=step_per_epoch,
        max_epoch=250,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--classnum', help='10 for cifar10 or 100 for cifar100',
                        type=int, default=10)
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    tp.logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]), 'd')

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config(args.classnum)
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        tp.QueueInputTrainer(config).train()
        #SimpleTrainer(config).train()
