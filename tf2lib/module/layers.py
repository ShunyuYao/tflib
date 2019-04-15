import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tfa.layers.LayerNormalization


class Pad(keras.layers.Layer):

    def __init__(self, paddings, mode='CONSTANT', constant_values=0, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode=self.mode, constant_values=self.constant_values)


class ReflectionPad2d(keras.layers.Layer):
    def __init__(self, paddings, **kwargs):
        super(ReflectionPad2d, self).__init__(name='ReflectionPad2d', **kwargs)
        self.paddings = paddings

    def call(self, x):
        x = tf.pad(x, self.paddings, 'REFLECT')
        return x


class Tanh(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(name='Tanh', **kwargs)

    def call(self, x):
        return keras.activations.tanh(x)


class Sigmoid(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(name='Sigmoid', **kwargs)

    def call(self, x):
        return keras.activations.sigmoid(x)
