from models.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input

class UnetLayer(tf.keras.layers.Layer):
    def __init__(self, feature, levels):
        super(UnetLayer, self).__init__()

        self.inputBlock = inputBlock(feature)
        self.downBlocks = [ downBLock(feature,level) for level in range(1,levels)  ]
        self.blottleBlock = blottleBlock(feature, levels)
        self.upBlocks = [ upBlock(feature,levels - level) for level in range(1,levels)]
        self.outBlock = outBlock(feature)

    def call(self, inputs):

        outputs = []
        x = self.inputBlock(inputs)
        outputs.append(x)

        for downlayer in self.downBlocks:
            x = downlayer(x)
            outputs.append(x)

        x = self.blottleBlock(x)    

        for uplayer in self.upBlocks:
            y = outputs.pop(-1)
            x = uplayer([x,y])

        y = outputs.pop(-1)
        x = self.outBlock([x,y])

        return x

def Unet(input_shape, feature=8, levels=3, num_classes=3):

    outputs = []

    _input = Input(input_shape)

    unet = UnetLayer(feature, levels)(_input)    

    _output = Conv2D(num_classes, 1, activation='softmax', padding="same")(unet)

    model = Model(_input, _output, name="Unet")
    return model