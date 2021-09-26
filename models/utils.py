import tensorflow as tf
from tensorflow.keras.metrics import mean_squared_error

def discriminator_loss(real_output, fake_output):
    loss = tf.math.log(real_output) + tf.math.log(1 - fake_output)
    return - tf.math.reduce_mean(loss)
    
def autoencoder_loss(real_x, estimated_x, real_output, fake_output, tau=0.001):
    autoencoder_loss = mean_squared_error(real_x, estimated_x)
    gan_loss =  tau*discriminator_loss(real_output, fake_output)
    total_loss = autoencoder_loss - gan_loss
    return total_loss
