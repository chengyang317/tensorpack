import tensorflow as tf
import tensorpack as tp
import argparse

BATCH_SIZE = 4


class Model(tp.ModelDesc):
    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, (None,), 'input'), tp.InputVar(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs, is_training):
        is_training = bool(is_training)
        image, label = inputs
        with tp.argscope(tp.Conv2D, kernel_shape=3, w_init_config=tp.FillerConfig(type='msra')):
            l = tp.Conv2D('conv1_1', image, 64)
            l = tp.Conv2D('conv1_2', l, 64)
            l = tp.MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            l = MaxPooling('pool2', l, 2)
            # 56

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            l = MaxPooling('pool3', l, 2)
            # 28

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            l = MaxPooling('pool4', l, 2)
            # 14

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            l = MaxPooling('pool5', l, 2)
        # 7

        l = FullyConnected('fc6', l, 4096)
        l = tf.nn.dropout(l, keep_prob)
        l = FullyConnected('fc7', l, 4096)
        l = tf.nn.dropout(l, keep_prob)
        logits = FullyConnected('fc8', l, out_dim=1000, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')


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