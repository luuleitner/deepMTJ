import glob
import os

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Lambda, Activation, Flatten, Reshape
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD

def distance(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(K.sqrt(dx * dx + dy * dy))

def correct(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(K.sqrt(dx * dx + dy * dy) < 8.08)

def squared_distance(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(K.abs(dx * dx + dy * dy))

def normed_squared_distance(y_true, y_pred):
    c = K.constant([2, 1], dtype=K.floatx())
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.sum(K.square((y_pred - y_true) * c), axis=-1) / K.constant(5., dtype=K.floatx())

def normed_rms_distance(y_true, y_pred):
    c = K.constant(143.1, dtype=K.floatx())
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(K.sqrt(dx * dx + dy * dy)) / c

def mse(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(dx * dx + dy * dy)

def std(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    distance = K.sqrt(dx * dx + dy * dy)
    return K.std(distance)

class AttentionVGG:

    def backbone(self, x, regularizer=None):
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv5')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv6')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv7')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        local1 = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv8')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv9')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv10')(x)
        x = BatchNormalization()(x)
        local2 = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(local2)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv11')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv12')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv13')(x)
        x = BatchNormalization()(x)
        local3 = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(local3)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv14')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv15')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
        x = Flatten(name='pregflatten')(x)
        g = Dense(512, activation='relu', kernel_regularizer=regularizer, name='globalg')(x)  # batch*512
        return (g, local1, local2, local3)

    def __init__(self, att='att3', compatibilityfunction='pc', height=32,
                 width=32, channels=3, weight_decay=0.0005,
                 optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0000001), loss=normed_squared_distance,
                 metrics=None):

        metrics = ['mae', distance, correct, mse] if metrics is None else metrics

        inp = Input(shape=(height, width, channels))
        input = inp
        regularizer = keras.regularizers.l2(weight_decay)

        (g, local1, local2, local3) = self.backbone(input, regularizer)

        l1 = Dense(512, kernel_regularizer=regularizer, name='l1connectordense')(local1)
        if compatibilityfunction == 'pc':
            c1 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc1')([l1, g])
        else:
            c1 = Lambda(lambda lam: K.squeeze(
                K.map_fn(lambda xy: K.dot(xy[0], xy[1]), elems=(lam[0], K.expand_dims(lam[1], -1)), dtype='float32'),
                3), name='cdp1')([l1, g])  # batch*x*y
        flatc1 = Flatten(name='flatc1')(c1)  # batch*xy
        a1 = Activation('softmax', name='softmax1')(flatc1)  # batch*xy
        reshaped1 = Reshape((-1, 512), name='reshape1')(l1)  # batch*xy*512.
        g1 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g1')(
            [a1, reshaped1])  # batch*512.

        l2 = local2
        if compatibilityfunction == 'pc':
            c2 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc2')([l2, g])
        else:
            c2 = Lambda(lambda lam: K.squeeze(
                K.map_fn(lambda xy: K.dot(xy[0], xy[1]), elems=(lam[0], K.expand_dims(lam[1], -1)), dtype='float32'),
                3), name='cdp2')([l2, g])
        flatc2 = Flatten(name='flatc2')(c2)
        a2 = Activation('softmax', name='softmax2')(flatc2)
        reshaped2 = Reshape((-1, 512), name='reshape2')(l2)
        g2 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g2')([a2, reshaped2])

        l3 = local3
        if compatibilityfunction == 'pc':
            c3 = ParametrisedCompatibility(kernel_regularizer=regularizer, name='cpc3')([l3, g])
        else:
            c3 = Lambda(lambda lam: K.squeeze(
                K.map_fn(lambda xy: K.dot(xy[0], xy[1]), elems=(lam[0], K.expand_dims(lam[1], -1)), dtype='float32'),
                3), name='cdp3')([l3, g])
        flatc3 = Flatten(name='flatc3')(c3)
        a3 = Activation('softmax', name='softmax3')(flatc3)
        reshaped3 = Reshape((-1, 512), name='reshape3')(l3)
        g3 = Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0], 1), lam[1]), 1), name='g3')([a3, reshaped3])

        glist = [g3]
        if att == 'att2':
            glist.append(g2)
        if att == 'att3':
            glist.append(g2)
            glist.append(g1)
        predictedG = g3
        if att != 'att1' and att != 'att':
            predictedG = Concatenate(axis=1, name='ConcatG')(glist)
        x = Dense(2, kernel_regularizer=regularizer, name=str(2) + 'ConcatG')(predictedG)
        out = Activation("sigmoid", name='concatsigmoid')(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = model

        outs = [Reshape((16, 32))(a3)] if att == 'att' or att == 'att1' else [Reshape((16, 32))(a3), Reshape((32, 64))(a2)] if att == 'att2' else [Reshape((16, 32))(a3), Reshape((32, 64))(a2), Reshape((64, 128))(a1)]
        self.attention_map_model = Model(inp, outs)

    def loadWeights(self, base_path):
        os.makedirs(os.path.join(base_path, 'weights'), exist_ok=True)
        pastepochs = [int(os.path.basename(file).replace('.hdf5', ''))
                      for file in glob.glob(os.path.join(base_path, 'weights', '*.hdf5'))]

        start_epoch = 0
        if len(pastepochs):
            start_epoch = max(pastepochs)
            self.model.load_weights(os.path.join(base_path, 'weights', '%d.hdf5') % start_epoch)
        return start_epoch


class ParametrisedCompatibility(Layer):

    def __init__(self, kernel_regularizer=None, **kwargs):
        super(ParametrisedCompatibility, self).__init__(**kwargs)
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        self.u = self.add_weight(name='u', shape=(input_shape[0][3], 1), initializer='uniform',
                                 regularizer=self.regularizer, trainable=True)
        super(ParametrisedCompatibility, self).build(input_shape)

    def call(self, x):  # add l and g. Dot the sum with u.
        return K.dot(K.map_fn(lambda lam: (lam[0] + lam[1]), elems=(x), dtype='float32'), self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
