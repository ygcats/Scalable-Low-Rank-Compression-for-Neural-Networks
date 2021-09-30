import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input
from utils import update_dict as _U
from model.layers import fc_bn_relu, fc, conv_bn_relu, max_pooling, flatten

# Default values
cfg = {
    'num_nodes': None,
    'num_filters': None,
    'kernel_size': 3,
    'kernel_stride': 1,
    'kernel_dilation': 1,
    'channel_order': 'channels_first',    # or 'channels_last'
    'padding_policy': 'same',             # or 'valid'
    'weight_init': tf.keras.initializers.VarianceScaling(   # fan_in & truncated_normal = tf.initializers.he_normal()
                    scale=2., mode="fan_in", distribution="truncated_normal", seed=None),
    'weight_regl': tf.keras.regularizers.l2(0.5),           # Keras ver. does not multiply 1/2
    'bias_regl': None,
    'bn_momentum': 0.9,
    'bn_epsilon': 1.0e-5,
    'bn_beta_reg': None,
    'bn_gamma_reg': None
}


class vgg15(tf.keras.Model):
    def __init__(self, nr_out_nodes, name):
        super(vgg15, self).__init__(name=name)

        self.layer = [
            conv_bn_relu(_U(cfg, {'num_filters': 64}), name='CBR1'), Dropout(rate=0.3, name='drp1'),
            conv_bn_relu(_U(cfg, {'num_filters': 64}), name='CBR2'),
            max_pooling(_U(cfg, {'kernel_size': 2, 'kernel_stride': 2}), name='mp1'),

            conv_bn_relu(_U(cfg, {'num_filters': 128}), name='CBR3'), Dropout(rate=0.4, name='drp2'),
            conv_bn_relu(_U(cfg, {'num_filters': 128}), name='CBR4'),
            max_pooling(_U(cfg, {'kernel_size': 2, 'kernel_stride': 2}), name='mp2'),

            conv_bn_relu(_U(cfg, {'num_filters': 256}), name='CBR5'), Dropout(rate=0.4, name='drp3'),
            conv_bn_relu(_U(cfg, {'num_filters': 256}), name='CBR6'), Dropout(rate=0.4, name='drp4'),
            conv_bn_relu(_U(cfg, {'num_filters': 256}), name='CBR7'),
            max_pooling(_U(cfg, {'kernel_size': 2, 'kernel_stride': 2}), name='mp3'),

            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR8'), Dropout(rate=0.4, name='drp5'),
            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR9'), Dropout(rate=0.4, name='drp6'),
            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR10'),
            max_pooling(_U(cfg, {'kernel_size': 2, 'kernel_stride': 2}), name='mp4'),

            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR11'), Dropout(rate=0.4, name='drp7'),
            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR12'), Dropout(rate=0.4, name='drp8'),
            conv_bn_relu(_U(cfg, {'num_filters': 512}), name='CBR13'),
            max_pooling(_U(cfg, {'kernel_size': 2, 'kernel_stride': 2}), name='mp5'),
            
            flatten(cfg, name='flatten'),
            Dropout(rate=0.5, name='drp9'),
            fc_bn_relu(_U(cfg, {'num_nodes': 512}), name='FBR1'),
            Dropout(rate=0.5, name='drp10'),
            fc(_U(cfg, {'num_nodes': nr_out_nodes}), name='fc_last')
        ]

    def call(self, inputs, training=False):
        h = Input(tensor=inputs)
        for l in self.layer:
            h = l(h, training=training)
        return h
