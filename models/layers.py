import tensorflow as tf
from tensorflow.keras.layers import *


FACTOR = [1 , 1 , 1/2 , 1/2 , 1/4 ]

class convBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64, batch_norm=True, activation='relu'):
        super(convBlock, self).__init__()

        conv = Conv2D(filters, 3, activation=None, padding="same")
        activation = Activation(activation)

        self.layers = [conv, activation]

        if batch_norm:
            bn = BatchNormalization()
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
        for factor in factors:
            feature = int(factor*features)
            conv = convBlock(feature)
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
        
        conv = convBlock(out_dim, batch_norm=False)
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
        self.last_conv = convBlock(1, batch_norm=False, activation='sigmoid')

    def call(self, inputs):

        x = self.encoder(inputs)
        x = self.last_conv(x)

        return x



