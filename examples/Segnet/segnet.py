import argparse
import os
import tensorflow as tf
import tensorpack as tp
import numpy as np

BATCH_SIZE = 4


class Model(tp.ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, (None, 500, 500, 3), 'input'), tp.InputVar(tf.int32, (None, 500, 500), 'label')]

    def _build_graph(self, inputs, is_training):
        is_training = bool(is_training)
        images, labels = inputs
        with tp.argscope.scope(tp.Conv2D, kernel_shape=3, w_init_config=tp.FillerConfig(type='msra'),
                               nl=tp.BNReLU(is_training)):
            l = tp.Conv2D('conv1_1', images, 64)
            l = tp.Conv2D('conv1_2', l, 64)
            l, pool1_mask = tp.MaxPoolingWithArgmax('pool1', l, 2)
            l = tp.Conv2D('conv2_1', l, 128)
            l = tp.Conv2D('conv2_2', l, 128)
            l, pool2_mask = tp.MaxPoolingWithArgmax('pool2', l, 2)
            l = tp.Conv2D('conv3_1', l, 256)
            l = tp.Conv2D('conv3_2', l, 256)
            l = tp.Conv2D('conv3_3', l, 256)
            l, pool3_mask = tp.MaxPoolingWithArgmax('pool3', l, 2)
            l = tp.Conv2D('conv4_1', l, 512)
            l = tp.Conv2D('conv4_2', l, 512)
            l = tp.Conv2D('conv4_3', l, 512)
            l, pool4_mask = tp.MaxPoolingWithArgmax('pool4', l, 2)
            l = tp.Conv2D('conv5_1', l, 512)
            l = tp.Conv2D('conv5_2', l, 512)
            l = tp.Conv2D('conv5_3', l, 512)
            l, pool5_mask = tp.MaxPoolingWithArgmax('pool5', l, 2)

            l = tp.ArgmaxUnPooling('upsample5', l, pool5_mask, shape_scale=2, stride=2)
            l = tp.Conv2D('conv5_3_D', l, 512)
            l = tp.Conv2D('conv5_2_D', l, 512)
            l = tp.Conv2D('conv5_1_D', l, 512)
            l = tp.ArgmaxUnPooling('upsample4', l, pool4_mask, shape_scale=2, stride=2)
            l = tp.Conv2D('conv4_3_D', l, 512)
            l = tp.Conv2D('conv4_2_D', l, 512)
            l = tp.Conv2D('conv4_1_D', l, 256)
            l = tp.ArgmaxUnPooling('upsample3', l, pool3_mask, shape_scale=2, stride=2)
            l = tp.Conv2D('conv3_3_D', l, 256)
            l = tp.Conv2D('conv3_2_D', l, 256)
            l = tp.Conv2D('conv3_1_D', l, 128)
            l = tp.ArgmaxUnPooling('upsample2', l, pool2_mask, shape_scale=2, stride=2)
            l = tp.Conv2D('conv2_2_D', l, 128)
            l = tp.Conv2D('conv2_1_D', l, 64)
            l = tp.ArgmaxUnPooling('upsample1', l, pool1_mask, shape_scale=2, stride=2)
            l = tp.Conv2D('conv1_2_D', l, 64)
            logits = tp.Conv2D('conv1_1_D', l, 21)

            segm_loss = tp.segm_loss('segm_loss', logits, labels, keys=tp.MOVING_SUMMARY_VARS_KEY)
            segm_accuracy = tp.segm_pixel_accuracy('segm_accuracy', logits, labels, keys=tp.MOVING_SUMMARY_VARS_KEY)
            wd_loss = tp.regularize_loss('regularize_loss', 'conv.*/W', 0.0005, keys=tp.MOVING_SUMMARY_VARS_KEY)

            tp.add_param_summary([('.*/W', ['histogram'])])  # monitor W
            self.loss = tp.sum_loss('sum_loss', [segm_loss, wd_loss])
            return locals()


def get_data(train_or_test):
    is_train = train_or_test == 'train'
    ds = tp.dataset.VOC12Seg(dataset_type=train_or_test, shuffle=True)
    augmentors = [tp.imgaug.Padding(target_shape=500)]
    if is_train:
        ds = tp.AugmentImagesTogether(ds, augmentors)
    else:
        ds = tp.AugmentImageComponent(ds, augmentors)
    ds = tp.BatchData(ds, BATCH_SIZE, remainder=not is_train)
    ds = tp.PrefetchDataZMQ(ds, 5)
    return ds


def get_config():
    dataset_train = get_data('train')
    # step_per_epoch = dataset_train.size()
    step_per_epoch = 5
    dataset_test = get_data('val')
    sess_config = tp.get_default_sess_config(0.5)
    nr_gpu = tp.get_nr_gpu()
    lr = tf.train.exponential_decay(learning_rate=1e-2, global_step=tp.get_global_step_var(),
                                    decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
                                    decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)
    return tp.TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=tp.Callbacks([tp.StatPrinter(), tp.ModelSaver(),
                                tp.InferenceRunner(dataset_test, tp.SegmAccuracy(output_name='segm_accuracy'))]),
        session_config=sess_config,
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=250,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    basename = os.path.basename(__file__)
    tp.logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]), 'd')

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with tf.Graph().as_default():
        config = get_config()
        if args.load:
            config.session_init = tp.SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        tp.QueueInputTrainer(config).train()