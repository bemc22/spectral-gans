import tensorflow as tf


def PSNR( max_val=1):

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=max_val)

    return  psnr


def SSIM( max_val=1):

    def ssim(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=max_val)

    return  ssim