import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Dense

def get_conv_block(filters, kernel_size = 3, strides = 2, padding = 'same'):
    conv = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)
    bn = BatchNormalization()
    relu = ReLU()
    return Sequential([conv, bn, relu])

def get_model(input_shape):
    # 32 x 32 x3
    input = Input(shape = input_shape)
    # 32 x 32 x 3 -> 16 x 16 x 32
    out = get_conv_block(32)(input)
    # 16 x 16 x 32 -> 8 x 8 x 64
    out = get_conv_block(64)(out)
    # 8 x 8 x 64 -> 4 x 4 x 64
    out = get_conv_block(64)(out)
    # 4 x 4 x 64 -> 2 x 2 x 64
    out = AveragePooling2D(pool_size = (2, 2))(out)
    out = Flatten()(out)
    out = Dense(units = 10)(out)
    return Model(inputs = input, outputs = out)


