import argparse

import tensorflow as tf

import tensorpack as tp

BATCH_SIZE = 4


class Model(tp.ModelDesc):
    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, (None,), 'input'), tp.InputVar(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs, is_training):
        is_training = bool(is_training)
        image, label = inputs
        with tp.argscope(tp.Conv2D, kernel_shape=3, w_init_config=tp.FillerConfig(type='msra'),
                         nl=tp.BNReLU(is_training)):
            l = tp.Conv2D('conv1_1', image, 64)
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

            l = tp.ArgmaxUnPooling('upsample5', x=l, argmax=pool5_mask, shape=2, stride=2)
            l = tp.Conv2D('conv5_3_D', l, 512)
            l = tp.Conv2D('conv5_2_D', l, 512)
            l = tp.Conv2D('conv5_1_D', l, 512)
            l = tp.ArgmaxUnPooling('upsample4', x=l, argmax=pool4_mask, shape=2, stride=2)
            l = tp.Conv2D('conv4_3_D', l, 512)
            l = tp.Conv2D('conv4_2_D', l, 512)
            l = tp.Conv2D('conv4_1_D', l, 256)
            l = tp.ArgmaxUnPooling('upsample3', x=l, argmax=pool3_mask, shape=2, stride=2)
            l = tp.Conv2D('conv3_3_D', l, 256)
            l = tp.Conv2D('conv3_2_D', l, 256)
            l = tp.Conv2D('conv3_1_D', l, 128)
            l = tp.ArgmaxUnPooling('upsample2', x=l, argmax=pool2_mask, shape=2, stride=2)
            l = tp.Conv2D('conv2_2_D', l, 128)
            l = tp.Conv2D('conv2_1_D', l, 64)
            l = tp.ArgmaxUnPooling('upsample1', x=l, argmax=pool1_mask, shape=2, stride=2)
            l = tp.Conv2D('conv1_2_D', l, 64)
            logits = tp.Conv2D('conv1_1_D', l, 11)

            pos_weight = [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614]
            cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
            cost = tf.reduce_mean(cost, name='cross_entropy_loss')
            tf.add_to_collection(tp.MOVING_SUMMARY_VARS_KEY, cost)

            # compute the number of failed samples, for ClassificationError to use at test time
            wrong = tp.prediction_incorrect(logits, label)
            nr_wrong = tf.reduce_sum(wrong, name='wrong')
            # monitor training error
            tf.add_to_collection(tp.MOVING_SUMMARY_VARS_KEY, tf.reduce_mean(wrong, name='train_error'))

            # weight decay on all W of fc layers
            wd_cost = tf.mul(0.0005, tp.regularize_cost('conv.*/W', tf.nn.l2_loss), name='regularize_loss')
            tf.add_to_collection(tp.MOVING_SUMMARY_VARS_KEY, wd_cost)

            tp.add_param_summary([('.*/W', ['histogram'])])  # monitor W
            self.cost = tf.add_n([cost, wd_cost], name='cost')


def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((30, 30)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2,1.8)),
            imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (30,30), 0.2, 3),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds

def get_config(cifar_classnum):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    # step_per_epoch = dataset_train.size()
    step_per_epoch = 5
    dataset_test = get_data('test', cifar_classnum)

    sess_config = get_default_sess_config(0.5)

    nr_gpu = get_nr_gpu()
    lr = tf.train.exponential_decay(
        learning_rate=1e-2,
        global_step=get_global_step_var(),
        decay_steps=step_per_epoch * (30 if nr_gpu == 1 else 20),
        decay_rate=0.5, staircase=True, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            InferenceRunner(dataset_test, ClassificationError())
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
    logger.set_logger_dir(
        os.path.join('train_log', basename[:basename.rfind('.')]), 'k')

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
        QueueInputTrainer(config).train()