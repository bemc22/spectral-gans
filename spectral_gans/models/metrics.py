import tensorflow as tf
import math

def PSNR( max_val=1):

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=max_val)

    return  psnr


def SSIM( max_val=1):

    def ssim(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=max_val)

    return  ssim
 
def SAM(y_true,y_pred):

    pp = tf.sqrt(tf.reduce_sum(tf.pow(y_true,2), 2, keepdims=True))+ 2.2e-16
    pp2 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred,2), 2, keepdims=True)) + 2.2e-16
    y_true = tf.divide(y_true,pp)
    y_pred = tf.divide(y_pred,pp2)
    z = tf.reduce_sum(tf.multiply(y_true,y_pred),2)
    z = tf.reduce_mean(tf.acos(z-2.2e-16)*180/ math.pi)
    return z

def ACCURACY(value):

    threshold = 0.5

    def accuracy(y_pred):
        
        y_true = tf.constant( value, shape=y_pred.shape, dtype=tf.float32)
        y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
        tp = tf.reduce_sum( y_true*y_pred )
        tn = tf.reduce_sum(  (1-y_true)*(1-y_pred) )
        acc = ( tp + tn) / tf.size(y_pred, out_type=tf.float32)

        return acc

    return accuracy
        

