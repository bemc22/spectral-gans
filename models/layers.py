import tensorflow as tf
from tensorflow.keras.layers import *

class inputBlock(tf.keras.layers.Layer):
    def __init__(self, F):
        super(inputBlock, self).__init__()

        self.conv1 = Conv2D(F, 3,activation='relu',padding="same")
        self.conv2 = Conv2D(F, 3,activation='relu',padding="same")
        
    def call(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(inputs)        

        return outputs


class downBLock(tf.keras.layers.Layer):
    def __init__(self, F, lvl):
        super(downBLock, self).__init__()

        feature = F*(2**lvl)
        self.down  = MaxPooling2D(pool_size=(2, 2))
        self.conv1 = Conv2D(feature, 3,activation='relu',padding="same")
        self.conv2 = Conv2D(feature, 3,activation='relu',padding="same")
        
    def call(self, inputs):

        outputs = self.down(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)        

        return outputs

class blottleBlock(tf.keras.layers.Layer):
    def __init__(self, F, lvl):
        super(blottleBlock, self).__init__()

        feature = F*(2**lvl)
        self.down  = MaxPooling2D(pool_size=(2, 2))
        self.conv1 = Conv2D(feature, 3,activation='relu',padding="same")
        self.conv2 = Conv2D(int(feature/2), 3,activation='relu',padding="same")
        
    def call(self, inputs):

        outputs = self.down(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)        

        return outputs

class upBlock(tf.keras.layers.Layer):
    def __init__(self, F, lvl):
        super(upBlock, self).__init__()

        feature = F*(2**lvl)
        self.up     = UpSampling2D(size=(2, 2))
        self.concat = Concatenate()
        self.conv1  = Conv2D(feature, 3,activation='relu',padding="same")
        self.conv2  = Conv2D(int(feature/2), 3,activation='relu',padding="same")        

    def call(self, inputs):

        outputs = self.up(inputs[0])
        outputs = self.concat([outputs,inputs[1]])
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs

class outBlock(tf.keras.layers.Layer):
    def __init__(self, F):
        super(outBlock, self).__init__()

        self.up     = UpSampling2D(size=(2, 2))
        self.concat = Concatenate()
        self.conv1  = Conv2D(F, 3,activation='relu',padding="same")
        self.conv2  = Conv2D(F, 3,activation='relu',padding="same")        

    def call(self, inputs):

        outputs = self.up(inputs[0])
        outputs = self.concat([outputs,inputs[1]])
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs
