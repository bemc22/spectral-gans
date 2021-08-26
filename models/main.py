import tensorflow as tf

from models.layers import Encoder, Generator, Discriminator, FACTOR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import Loss , MSE

class loss1(Loss):

    def call(self, y_true, y_pred):

        loss = tf.math.log(y_pred)
        return - tf.reduce_mean(loss)

class loss2(Loss):

    def call(self, y_true, y_pred):

        loss = tf.math.log(1 - y_pred)
        return - tf.reduce_mean(loss)


def SpectralGAN(input_shape, features=64, factors=FACTOR):

    out_dim = input_shape[-1]
    _input = Input(input_shape)

    encoder = Encoder(features=features, factors=FACTOR)
    generator = Generator(out_dim, features=features, factors=FACTOR)
    discriminator = Discriminator(features=features, factors=FACTOR)

    z = encoder(_input)
    _output = generator(z)

    real = discriminator(_input)
    fake = discriminator(_output)


    model = Model( _input , [_output, real, fake] , name='SpectralGAN' )
    return model









    