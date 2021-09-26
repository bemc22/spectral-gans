import tensorflow as tf


def PSNR( max_val=1):

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=max_val)

    return  psnr


def SSIM( max_val=1):

    def ssim(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=max_val)

    return  ssim


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
        

