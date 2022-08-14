"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers

	Copyright (C) 2021 by the authors: Jarolim Robert (University of Graz), <robert.jarolim@uni-graz.at> and
	Leitner Christoph (Graz University of Technology), <christoph.leitner@tugraz.at>.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, \
    add, multiply, ReLU
from keras.layers import core, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.models import Model


def up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_last':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))

    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, skip, data_format='channels_last'):
    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    skip, att_map = attention_block_2d(x=skip, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_last':
        concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))

    x = concat([up, skip])
    return x, att_map


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    att_map = Activation('sigmoid')(psi_f)

    # att_map(?,x_height,x_width)

    # x(?,x_height,x_width,x_channel)

    x = multiply([x, att_map])

    return x, att_map


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

              padding='same', data_format='channels_last'):
    if data_format == 'channels_last':
        input_n_filters = input_layer.get_shape().as_list()[3]
    else:
        input_n_filters = input_layer.get_shape().as_list()[1]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_last'):
    if data_format == 'channels_last':
        input_n_filters = input_layer.get_shape().as_list()[3]
    else:
        input_n_filters = input_layer.get_shape().as_list()[1]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


########################################################################################################
# Define the neural network
def unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        # attention_up_and_concate(x,[skips[i])
        x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
        x = concatenate([skips[i], x], axis=1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    # model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
# Attention U-Net
def att_unet(img_h, img_w, channels, n_label, data_format='channels_last'):
    inputs = Input((img_h, img_w, channels))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.)(x)
        x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.)(x)
    x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    att_maps = []
    for i in reversed(range(depth)):
        features = features // 2
        x, att_map = attention_up_and_concate(x, skips[i], data_format=data_format)
        att_maps.append(att_map)
        x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.)(x)
        x = Conv2D(features, (3, 3), padding='same', data_format=data_format)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    out = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=out)
    att_model = Model(inputs=inputs, outputs=[out, *att_maps])

    return model, att_model


########################################################################################################
# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model


########################################################################################################
# Attention R2U-Net
def att_r2_unet(img_w, img_h, n_label, data_format='channels_last'):
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model
