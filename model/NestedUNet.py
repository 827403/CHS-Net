import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model

#函数式模块实现
def conv_block_nested(x, mid_ch, out_ch, kernel_size=3, padding='same'):
    x = Conv2D(mid_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(out_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def NestedUNet(nClasses=1, input_height=256, input_width=256):
     
    inputs = Input(shape=(input_height, input_width, 3))
    t = 2
    n1 = 16
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    x0_0 = conv_block_nested(inputs, filters[0], filters[0])

    x1_0 = conv_block_nested(MaxPooling2D(strides=2)(x0_0), filters[1], filters[1])
    x0_1 = conv_block_nested(Concatenate()([x0_0, UpSampling2D()(x1_0)]), filters[0], filters[0])

    x2_0 = conv_block_nested(MaxPooling2D(strides=2)(x1_0), filters[2], filters[2])
    x1_1 = conv_block_nested(Concatenate()([x1_0, UpSampling2D()(x2_0)]), filters[1], filters[1])
    x0_2 = conv_block_nested(Concatenate()([x0_0, x0_1, UpSampling2D()(x1_1)]), filters[0], filters[0])

    x3_0 = conv_block_nested(MaxPooling2D(strides=2)(x2_0), filters[3], filters[3])
    x2_1 = conv_block_nested(Concatenate()([x2_0, UpSampling2D()(x3_0)]), filters[2], filters[2])
    x1_2 = conv_block_nested(Concatenate()([x1_0, x1_1, UpSampling2D()(x2_1)]), filters[1], filters[1])
    x0_3 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, UpSampling2D()(x1_2)]), filters[0], filters[0])

    x4_0 = conv_block_nested(MaxPooling2D(strides=2)(x3_0), filters[4], filters[4])
    x3_1 = conv_block_nested(Concatenate()([x3_0, UpSampling2D()(x4_0)]), filters[3], filters[3])
    x2_2 = conv_block_nested(Concatenate()([x2_0, x2_1, UpSampling2D()(x3_1)]), filters[2], filters[2])
    x1_3 = conv_block_nested(Concatenate()([x1_0, x1_1, x1_2, UpSampling2D()(x2_2)]), filters[1], filters[1])
    x0_4 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling2D()(x1_3)]), filters[0], filters[0])

    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(x0_4)

    model = Model(inputs=[inputs], outputs=[outputs])

#     model = Model(input=inputs, output=out)
#     model.outputHeight = outputHeight
#     model.outputWidth = outputWidth

    return model

