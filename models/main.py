import tensorflow as tf

from models.layers import Encoder, Generator, Discriminator, FACTOR
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import Loss , MSE



def make_autoencoder(input_shape, features=64, factors=FACTOR):
    out_dim = input_shape[-1]
    _input = Input(input_shape)
    encoder = Encoder(features=features, factors=FACTOR)
    generator = Generator(out_dim, features=features, factors=FACTOR)

    z = encoder(_input)    
    _output = generator(z)
    model = Model( _input , _output , name='generator' )
    return model

def make_discriminator(input_shape, features=64, factors=FACTOR):
    _input = Input(input_shape)
    discriminator = Discriminator(features=features, factors=FACTOR)
    _output = discriminator(_input)
    model = Model( _input , _output , name='discriminator' )
    return model










    