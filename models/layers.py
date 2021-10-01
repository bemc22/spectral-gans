import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2


FACTOR = [1/4, 1/4, 1/2, 1/2, 1]
NORMALIZE = False


class encodedLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(encodedLayer, self).__init__()

    def build(self, input_shape):
        self.alpha = self.add_weight("alpha", shape=input_shape[1:],
            initializer=tf.keras.initializers.glorot_uniform, 
            trainable=True)
    
    def call(self, inputs):
        return tf.multiply(inputs, self.alpha)



class convBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64, normalize=NORMALIZE, activation='relu'):
        super(convBlock, self).__init__()

        branch1 = Conv2D(filters, (3, 1),
                         activation=None,
                         padding="same",
                         kernel_regularizer=L2(1e-8))

        branch2 = Conv2D(filters, (1, 3),
                         activation=None,
                         padding="same",
                         kernel_regularizer=L2(1e-8))

        self.layers = []

        self.layers.append(Activation(activation))

        if normalize:
            self.layers.append(LayerNormalization())

        self.b1 = branch1
        self.b2 = branch2

    def call(self, inputs):

        x1 = self.b1(inputs)
        x2 = self.b2(inputs)
        x = x1 + x2

        for layer in self.layers:
            x = layer(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, features=64, factors=FACTOR):
        super(Encoder, self).__init__()

        self.convs = []

        for factor in factors[:-1]:
            feature = int(factor*features)
            conv = convBlock(feature)
            self.convs.append(conv)

        feature = int(factors[-1]*features)
        conv = convBlock(feature, activation=None, normalize=False)
        self.convs.append(conv)

    def call(self, inputs):

        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x


class Generator(tf.keras.layers.Layer):
    def __init__(self, out_dim, features=64, factors=FACTOR):
        super(Generator, self).__init__()

        self.convs = []

        for factor in factors[-2::-1]:
            feature = int(factor*features)
            conv = convBlock(feature)
            self.convs.append(conv)

        conv = convBlock(out_dim, activation='relu', normalize=False)
        self.convs.append(conv)

    def call(self, inputs):

        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x


class DownSample(tf.keras.layers.Layer):
    def __init__(self, filters, size, normalize=True):
        super(DownSample, self).__init__()
        self.layers = []

        initializer = tf.random_normal_initializer(0., 0.02)
        conv = Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False)

        self.layers.append(conv)
        if normalize:
            self.layers.append(BatchNormalization())
        self.layers.append(LeakyReLU())

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.layers = [
            Concatenate(),
            DownSample(64, 2, False),
            DownSample(128, 2),     
            #ZeroPadding2D(),       
            #BatchNormalization(),     
            #LeakyReLU(),
            #ZeroPadding2D(),       
            Conv2D(1, 2, strides=1, activation='sigmoid',
                   kernel_initializer=initializer)  
        ]

    def call(self, inputs):

        x = inputs

        for layer in self.layers:
            x = layer(x)

        return x
