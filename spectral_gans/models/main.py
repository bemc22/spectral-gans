import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.losses import Loss , MSE


from spectral_gans.models.layers import Encoder, Generator, Discriminator, encodedLayer , FACTOR
from spectral_gans.models.metrics import ACCURACY
from spectral_gans.models.utils import spacial_tv, spec2rgb


def make_generator(input_shape, features=64, factors=FACTOR):

    out_dim = input_shape[-1]
    in_dim = int(features*factors[-1])
    input_shape = input_shape[:-1] + (in_dim,)
    z = Input(input_shape)
    generator = Generator(out_dim, features=features, factors=factors)
    _output = generator(z)
    model = Model( z , _output , name='generator' )
    return model

def make_encoder(input_shape, features=64, factors=FACTOR):
    _input = Input(input_shape)
    encoder = Encoder(features=features, factors=factors)
    z = encoder(_input)    
    model = Model( _input , z , name='encoder' )
    return model

def make_discriminator(input_shape):
    target_shape = input_shape[:-1] + (3,)
    _input = Input(input_shape)
    _target = Input(target_shape)
    discriminator = Discriminator()
    _output = discriminator([_input, _target])
    model = Model( [_input, _target] , _output , name='discriminator' )

    return model


class spectralGAN(tf.keras.Model):
    def __init__(self, encoder, generator, discriminator, CMF):
        super(spectralGAN, self).__init__()

        self.encoder = encoder
        self.generator = generator

        self.autoencoder = tf.keras.Sequential(
            [self.encoder, self.generator]
        )

        self.spec2rgb = Lambda( lambda x:  spec2rgb(x, CMF) )
        self.discriminator = discriminator
        self.real_acc = ACCURACY(1)
        self.fake_acc = ACCURACY(0)

    def compile(self, a_optimizer, d_optimizer, a_loss, d_loss, metrics=[]):
        super(spectralGAN, self).compile(metrics=metrics)
        self.a_optimizer = a_optimizer
        self.d_optimizer = d_optimizer
        self.a_loss = a_loss
        self.d_loss = d_loss

    def call(self, inputs, training=None):

        target =  self.spec2rgb(inputs)
        generated = self.autoencoder(target, training=training)
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
        
        real_acc = self.real_acc(real_output)
        fake_acc = self.fake_acc(fake_output)

        resul['real_acc'] = real_acc
        resul['fake_acc'] = fake_acc

        return  resul

    def test_step(self, images):

        generated_images , real_output, fake_output = self(images, training=False)
        a_loss = self.a_loss(images, generated_images, real_output, fake_output)
        d_loss = self.d_loss(real_output, fake_output)

        self.compiled_metrics.update_state(images, generated_images)
        resul = {"a_loss": a_loss, "d_loss": d_loss}

        for m in self.metrics:
            resul[m.name] = m.result()

        real_acc = self.real_acc(real_output)
        fake_acc = self.fake_acc(fake_output)

        resul['real_acc'] = real_acc
        resul['fake_acc'] = fake_acc
        
        return resul