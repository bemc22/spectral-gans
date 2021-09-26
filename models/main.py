import tensorflow as tf

from models.layers import Encoder, Generator, Discriminator, FACTOR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import Loss , MSE


def make_autoencoder(input_shape, features=64, factors=FACTOR):
    out_dim = input_shape[-1]
    _input = Input(input_shape)
    encoder = Encoder(features=features, factors=factors)
    generator = Generator(out_dim, features=features, factors=factors)
    z = encoder(_input)    
    _output = generator(z)
    model = Model( _input , _output , name='generator' )
    return model

def make_discriminator(input_shape):
    target_shape = input_shape[:-1] + (1,)
    _input = Input(input_shape)
    _target = Input(target_shape)
    discriminator = Discriminator()
    _output = discriminator([_input, _target])
    model = Model( [_input, _target] , _output , name='discriminator' )

    return model


class spectralGAN(tf.keras.Model):
    def __init__(self, autoencoder, discriminator):
        super(spectralGAN, self).__init__()
        self.autoencoder = autoencoder
        self.discriminator = discriminator

    def compile(self, a_optimizer, d_optimizer, a_loss, d_loss, metrics=[]):
        super(spectralGAN, self).compile(metrics=metrics)
        self.a_optimizer = a_optimizer
        self.d_optimizer = d_optimizer
        self.a_loss = a_loss
        self.d_loss = d_loss

    def call(self, inputs, training=None):

        target = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        generated = self.autoencoder(inputs, training=training)
        real_output = self.discriminator([inputs, target], training=training)
        fake_output = self.discriminator([generated, target], training=training)

        return generated, real_output, fake_output

    def train_step(self, images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images , real_output, fake_output = self(images, training=True)
            
            # LOSS COMPUTING
            a_loss = self.a_loss(images, generated_images, real_output, fake_output)
            d_loss = self.d_loss(real_output, fake_output)

        # BACKPROPAGATION
        a_gradients = gen_tape.gradient(a_loss, self.autoencoder.trainable_variables)
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.a_optimizer.apply_gradients(zip(a_gradients, self.autoencoder.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        self.compiled_metrics.update_state(images, generated_images)

        resul = {"a_loss": a_loss, "d_loss": d_loss}

        for m in self.metrics:
            resul[m.name] = m.result()

        return  resul

    def test_step(self, images):

        generated_images , real_output, fake_output = self(images, training=False)
        a_loss = self.a_loss(images, generated_images, real_output, fake_output)
        d_loss = self.d_loss(real_output, fake_output)

        self.compiled_metrics.update_state(images, generated_images)
        resul = {"a_loss": a_loss, "d_loss": d_loss}

        for m in self.metrics:
            resul[m.name] = m.result()
        
        return resul







    