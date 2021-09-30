from tensorflow.keras.layers import Layer, Conv2D, Dense, BatchNormalization, ReLU, MaxPool2D, Flatten


class fc_bn_relu(Layer):
    def __init__(self, config, name=None):
        super(fc_bn_relu, self).__init__(name=name)
        self.dense = Dense(
            units=config['num_nodes'],
            kernel_initializer=config['weight_init'],
            kernel_regularizer=config['weight_regl'],
            use_bias=False,
            name='fc'
        )
        self.bn = BatchNormalization(
            momentum=config['bn_momentum'],
            epsilon=config['bn_epsilon'],
            axis=1,
            beta_regularizer=config['bn_beta_reg'],
            gamma_regularizer=config['bn_gamma_reg'],
            name='bn'
        )
        self.relu = ReLU(name='relu')

    def call(self, inputs, training=False):
        h = self.dense(inputs)
        h = self.bn(h, training=training)
        h = self.relu(h)
        return h


class fc(Layer):
    def __init__(self, config, name=None):
        super(fc, self).__init__(name=name)
        self.dense = Dense(
            units=config['num_nodes'],
            kernel_initializer=config['weight_init'],
            kernel_regularizer=config['weight_regl'],
            bias_regularizer=config['bias_regl'],
            name='fc'
        )

    def call(self, inputs, training=False):
        return self.dense(inputs)


class conv_bn_relu(Layer):
    def __init__(self, config, name=None):
        super(conv_bn_relu, self).__init__(name=name)
        self.conv = Conv2D(
            filters=config['num_filters'],
            kernel_size=config['kernel_size'],
            strides=config['kernel_stride'],
            padding=config['padding_policy'],
            data_format=config['channel_order'],
            dilation_rate=config['kernel_dilation'],
            kernel_initializer=config['weight_init'],
            kernel_regularizer=config['weight_regl'],
            use_bias=False,
            name='conv'
        )
        self.bn = BatchNormalization(
            momentum=config['bn_momentum'],
            epsilon=config['bn_epsilon'],
            axis=1 if config['channel_order'] == 'channels_first' else 3,
            beta_regularizer=config['bn_beta_reg'],
            gamma_regularizer=config['bn_gamma_reg'],
            name='bn'
        )
        self.relu = ReLU(name='relu')

    def call(self, inputs, training=False):
        h = self.conv(inputs)
        h = self.bn(h, training=training)
        h = self.relu(h)
        return h


class max_pooling(Layer):
    def __init__(self, config, name=None):
        super(max_pooling, self).__init__(name=name)
        self.max_pool = MaxPool2D(
            pool_size=config['kernel_size'],
            strides=config['kernel_stride'],
            padding=config['padding_policy'],
            data_format=config['channel_order'],
            name='mp'
        )

    def call(self, inputs, training=False):
        return self.max_pool(inputs)


class flatten(Layer):
    def __init__(self, config, name=None):
        super(flatten, self).__init__(name=name)
        self.flat = Flatten(
            data_format=config['channel_order'],
            name='flatten'
        )

    def call(self, inputs, training=False):
        return self.flat(inputs)
