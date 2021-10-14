import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.python.keras.backend import dtype

Phi = np.array([
    [ 0.41847, -0.15866, -0.082835],
    [-0.091169, 0.25243, 0.015708],
    [0.00092090, -0.0025498, 0.17860],             
]
)

def discriminator_loss(real_output, fake_output):
    loss = tf.math.log(real_output) + tf.math.log(1 - fake_output)
    return - tf.math.reduce_mean(loss)


def autoencoder_loss(tau=1e-4):
    def autoencoder_loss(real_x, estimated_x, real_output, fake_output):
        autoencoder_loss = tf.reduce_mean( tf.square(real_x - estimated_x) )
        gan_loss =  tau*discriminator_loss(real_output, fake_output)
        total_loss = autoencoder_loss - gan_loss
        return total_loss
        
    return autoencoder_loss

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

def g(x, alpha, mu, sigma1, sigma2):
  sigma = (x < mu)*sigma1 + ( x >= mu)*sigma2 
  return alpha*np.exp( (x-mu)**2 / (-2*(sigma**2) ) )

def get_RGB_matrix(start, stop, bands):
    color_r = lambda x: g(x, 1.056, 5998, 379, 310)  +  g(x, 0.362, 4420, 160, 267) +  g(x, -0.065, 5011, 204, 262)
    color_g = lambda x: g(x, 0.821, 5688, 469, 405)  +  g(x, 0.286, 5309, 163, 311)
    color_b = lambda x: g(x, 1.217, 4370, 118, 360)  +  g(x, 0.681, 4590, 260, 138)

    eje_x = np.linspace(start, stop, bands)*10
    CMF = np.array( [color_r(eje_x), color_g(eje_x), color_b(eje_x)] ).T
    RGB = CMF@Phi.T
    RGB = np.float32( RGB / np.sum(RGB, axis=0, keepdims=True) )
    return RGB