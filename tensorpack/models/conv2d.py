import tensorflow as tf
from tensorpack.models.layer import Layer, layer_class_register
from tensorpack.proto.caffe_pb2 import FillerParameter
from tensorpack.models.utils import shape2d, shape4d
from tensorpack.tfutils.variable import weight_create, bias_create

__all__ = ['Convolution']


class BaseConvolution(Layer):
    """"""

    @staticmethod
    def _layer_setup(input_tensors, layer_params):
        forward_params = {'x': input_tensors.values()[0]}
        assert layer_params.HasField('convolution_param')
        convolution_param = layer_params.convolution_param
        assert convolution_param.HasField('num_output')
        forward_params['num_output'] = convolution_param.num_outpweight_createut
        forward_params['kernel_size'] = convolution_param.kernel_size
        forward_params['group'] = convolution_param.group
        forward_params['weight_filler'] = convolution_param.weight_filler
        forward_params['bias_term'] = convolution_param.bias_term
        forward_params['bias_filler'] = convolution_param.bias_filler
        forward_params['stride'] = convolution_param.stride
        if convolution_param.HasField('stride_h') and convolution_param.HasField('stride_w'):
            forward_params['stride'] = (convolution_param.stride_h, convolution_param.stride_w)
        if convolution_param.HasField('pad') or convolution_param.HasField('pad_h'):
            forward_params['pad'] = 'VALID'
        else:
            forward_params['pad'] = 'SAME'
        return forward_params

    @staticmethod
    def _forward():
        pass


@layer_class_register.register('CONVOLUTION')
class Convolution(BaseConvolution):
    """"""

    @staticmethod
    def _forward(x, num_output, kernel_size, pad, stride=1, weight_filler=None, bias_filler=None, group=1,
                 bias_term=True):
        in_shape = x.get_shape().as_list()
        in_channel = in_shape[-1]
        assert in_channel % group == 0 and num_output % group == 0
        kernel_shape = shape2d(kernel_size)
        pad = pad.upper()
        filter_shape = kernel_shape + [in_channel / group, num_output]
        stride = shape4d(stride)
        if weight_filler is None:
            weight_filler = FillerParameter(type='xavier')
        if bias_term and not bias_filler:
            bias_filler = FillerParameter(type='constant')

        w_var = weight_create(weight_filler, filter_shape)
        if bias_term:
            bias_var = bias_create(bias_filler, [num_output])

        if group == 1:
            conv = tf.nn.conv2d(x, w_var, stride, pad)
        else:
            inputs = tf.split(3, group, x)
            kernels = tf.split(3, group, w_var)
            outputs = [tf.nn.conv2d(i, k, stride, pad) for i, k in zip(inputs, kernels)]
            conv = tf.concat(3, outputs)
        return tf.nn.bias_add(conv, bias_term) if bias_term else conv






































