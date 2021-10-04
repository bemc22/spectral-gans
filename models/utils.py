import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.python.keras.backend import dtype


def discriminator_loss(real_output, fake_output):
    loss = tf.math.log(real_output) + tf.math.log(1 - fake_output)
    return - tf.math.reduce_mean(loss)
    
def autoencoder_loss(real_x, estimated_x, real_output, fake_output, tau=1e-4):
    autoencoder_loss = mean_squared_error(real_x, estimated_x)
    gan_loss =  tau*discriminator_loss(real_output, fake_output)
    total_loss = autoencoder_loss - gan_loss
    return total_loss

def generator_loss(y_true, y_pred):
    loss = tf.math.log(y_pred)*y_true + tf.math.log( 1 - y_pred)*y_true
    return - tf.math.reduce_mean(loss)

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
    


def spec2rgb(x):
  (m,n, la) = np.shape(im)[1:]
  rgb = np.zeros((1, m, n, 3))
  rango = int(np.round(la/3))

  rgb[0, : , : , 2] = np.mean(x[0, :, :, 0:rango], axis = 2)
  rgb[0, : , : , 1] = np.mean(x[0, :, :, rango: 2*rango], axis = 2)
  rgb[0, : , : , 0] = np.mean(x[0, :, :, 2*rango:], axis = 2)
  return rgb