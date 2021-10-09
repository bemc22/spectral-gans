import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.python.keras.backend import dtype


def discriminator_loss(real_output, fake_output):
    loss = tf.math.log(real_output) + tf.math.log(1 - fake_output)
    return - tf.math.reduce_mean(loss)
    
def autoencoder_loss(real_x, estimated_x, real_output, fake_output, tau=1e-4):
    autoencoder_loss = tf.reduce_mean( tf.square(real_x - estimated_x) )
    gan_loss =  tau*discriminator_loss(real_output, fake_output)
    total_loss = autoencoder_loss - gan_loss
    return total_loss

def generator_loss(y_true, y_pred):
    loss = tf.log(y_pred)*y_true + tf.log( 1 - y_pred)*y_true
    return - tf.reduce_mean(loss)

@tf.function
def spacial_tv(inputs):
    dy, dx = tf.image.image_gradients(inputs)
    tv = tf.add(tf.abs(dy), tf.abs(dx))
    return tv

@tf.function
def soft_threshold(V, tau=1, ro=1):
    x = tau/ro
    V1 = tf.cast(V > x, dtype=tf.float32)*(V - x)
    V2 = tf.cast(V < -x, dtype=tf.float32)*(V + x)
    resul = V1 + V2
    return resul
    
@tf.function
def spec2rgb(_inputs, CMF):
  rgb = (_inputs @ CMF)
  return rgb
