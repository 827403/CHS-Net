#coding=utf-8
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def pool_block(inp, pool_factor):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]
    pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w)/ pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(int(x.shape[1])*strides[0], int(x.shape[2])*strides[1])))(x)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    return x

def PSPNet(nClasses, input_width=256, input_height=256):
#    assert input_height % 256 == 0
#    assert input_width % 256 == 0
    inputs = Input(shape=(input_height, input_width, 3))

    x = (Conv2D(64, (3, 3), activation='relu', padding='same'))(inputs)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f1 = x
    # 192 x 192

    x = (Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f2 = x
    # 96 x 96
    x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f3 = x
    # 48 x 48
    x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f4 = x

    # 24 x 24
    o = f4
    pool_factors = [1, 2, 4, 8]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(256, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    o = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(int(x.shape[1])*16, int(x.shape[2])*16)))(x)

    o = Conv2D(nClasses, (1, 1), padding='same')(o)
    o_shape = Model(inputs, o).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]
    print(outputHeight)
    print(outputWidth)
    o = (Reshape((outputHeight,outputWidth, nClasses)))(o)
    o = (Activation('sigmoid'))(o)
    model = Model(inputs=inputs, outputs=o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

