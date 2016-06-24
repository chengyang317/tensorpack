import tensorflow as tf
from tensorpack.proto.caffe_pb2 import FillerParameter
from tensorpack.models.layer import Layer, layer_class_register
from tensorpack.tfutils.symbolic_functions import batch_flatten
from tensorpack.tfutils.variable import weight_create, bias_create

__all__ = ['FullyConnected']


@layer_class_register.register('FullyConnected'.upper())
class FullyConnected(Layer):
    """"""
    @staticmethod
    def _layer_setup(input_tensors, layer_params):
        forward_params = {'x': input_tensors.values()[0]}
        assert layer_params.HasField('inner_product_param')
        inner_product_param = layer_params.inner_product_param
        assert inner_product_param.HasField('num_output')
        forward_params['num_output'] = inner_product_param.num_output
        forward_params['weight_filler'] = inner_product_param.weight_filler
        forward_params['bias_term'] = inner_product_param.bias_term
        forward_params['bias_filler'] = inner_product_param.bias_filler
        return forward_params

    @staticmethod
    def _forward(x, num_output, weight_filler=None, bias_filler=None, bias_term=True):
        x = batch_flatten(x)
        channel_dim = x.get_shape().as_list()[1]
        filter_shape = [channel_dim, num_output]
        if not weight_filler:
            weight_filler = FillerParameter(type='uniform_unit_scaling', factor=1.43)
        if bias_term and not bias_filler:
            bias_filler = FillerParameter(type='constant')
        w_var = weight_create(weight_filler, filter_shape)
        if bias_term:
            bias_var = bias_create(bias_filler, [num_output])
        prod = tf.nn.xw_plus_b(x, w_var, bias_var) if bias_term else tf.matmul(x, w_var)
        return prod

























