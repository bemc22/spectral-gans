import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import L2

FACTOR = [1, 1, 1/2, 1/2, 1/4]


class convBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64, normalize=False, activation='relu'):
        super(convBlock, self).__init__()

        conv = Conv2D(filters, 3,
                      activation=None,
                      padding="same",
                      kernel_regularizer=L2(1e-8))

        activation = Activation(activation)

        self.layers = [conv]
        self.layers.append(activation)

        if normalize:
            bn = LayerNormalization()
            self.layers.append(bn)

        

    def call(self, inputs):
        x = inputs
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


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, features=64, factors=FACTOR):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(features, factors)
        self.last_conv = convBlock(1, normalize=False, activation='sigmoid')

    def call(self, inputs):

        x = self.encoder(inputs)
        x = self.last_conv(x)
        return x
