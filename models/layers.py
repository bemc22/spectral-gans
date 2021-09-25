import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import L2

FACTOR = [1, 1, 1/2, 1/2, 1/4]
NORMALIZE = False


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


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, features=64, factors=FACTOR):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(features, factors)
        self.last_conv = convBlock(1, normalize=False, activation='sigmoid')

    def call(self, inputs):

        x = self.encoder(inputs)
        x = self.last_conv(x)
        return x


# Estructura del discriminador pix2pix
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def Discriminatorpix2pix():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[32, 32, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[32, 32, 3], name='target_image')

    # (batch_size, 32, 32, channels*2)
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)  # (batch_size, 16, 16, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 8, 8, 128)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)

    batchnorm1 = tf.keras.layers.BatchNormalization()(zero_pad1)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
