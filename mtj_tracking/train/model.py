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

import glob
import os

import numpy as np
from keras import backend as K
from keras.metrics import BinaryAccuracy, Recall
from tensorflow.keras.optimizers import SGD

###################################################################
## Loss Functions
from mtj_tracking.train.network import att_unet


def distance(y_true, y_pred):
    dist = []
    for y_t, y_p in zip(y_true, y_pred):
        pos_t = position(y_t)
        pos_p = position(y_p)
        dx = (pos_t[0] - pos_p[0]) * 2
        dy = (pos_t[1] - pos_p[1])
        dist.append(np.sqrt(dx * dx + dy * dy) / np.sqrt(5))
    return np.mean(dist)


def position(y):
    # return first position of max probability
    for pos in np.argwhere(y[..., 0] == np.max(y[..., 0])):
        return pos[1], pos[0]


def correct(y_true, y_pred):
    correct = []
    for y_t, y_p in zip(y_true, y_pred):
        pos_t = position(y_t)
        pos_p = position(y_p)
        dx = (pos_t[0] - pos_p[0])
        dy = (pos_t[1] - pos_p[1])
        correct.append((np.sqrt(dx * dx + dy * dy)) < 20)  # calculate in pixels
    return np.mean(correct)


def squared_distance(y_true, y_pred):
    dx = (y_true[..., 0] - y_pred[..., 0]) * 128
    dy = (y_true[..., 1] - y_pred[..., 1]) * 64

    return K.mean(K.abs(dx * dx + dy * dy))


def model_loss(y_true, y_pred):
    c_loss = class_loss(y_true, y_pred)
    return c_loss


def distance_loss(y_true, y_pred):
    d_loss = K.square(y_true[..., 1] - y_pred[..., 1]) + K.square(y_true[..., 2] - y_pred[..., 2])
    d_loss = d_loss * y_true[..., 0]  # ignore loss if not class 1
    d_loss = K.sum(d_loss) / (K.sum(y_true[..., 0]) + 1e-8)
    return d_loss


def class_loss(y_true, y_pred):
    norm = 10
    c_loss = K.binary_crossentropy(y_true[..., 0], y_pred[..., 0])
    c_loss = y_true[..., 0] * c_loss + (1 - y_true[..., 0]) * c_loss / norm
    c_loss = K.mean(c_loss)
    return c_loss


def model_accuracy(y_true, y_pred):
    return BinaryAccuracy()(y_true[..., 0], y_pred[..., 0])


def model_recall(y_true, y_pred):
    return Recall()(y_true[..., 0], y_pred[..., 0])


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
    print(distance)
    return K.std(distance)


###################################################################
class AttentionUNet:

    def __init__(self, height, width, channels=1,
                 optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0000001),
                 loss=model_loss,
                 metrics=[]):
        #
        model, attention_map_model = att_unet(height, width, channels, 1)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Generated Model")
        self.model = model
        self.attention_map_model = attention_map_model

    def loadWeights(self, base_path):
        os.makedirs(os.path.join(base_path, 'weights'), exist_ok=True)
        pastepochs = [int(os.path.basename(file).replace('.hdf5', ''))
                      for file in glob.glob(os.path.join(base_path, 'weights', '*.hdf5'))]

        start_epoch = 0
        if len(pastepochs):
            start_epoch = max(pastepochs)
            self.model.load_weights(os.path.join(base_path, 'weights', '%d.hdf5') % start_epoch)
        return start_epoch
