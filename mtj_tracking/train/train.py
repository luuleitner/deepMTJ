"""
	#deepMTJ
	an open source software tool made for biomechanical researchers

	Copyright (C) 2020 by the authors: Jarolim Robert (University of Graz), <robert.jarolim@uni-graz.at> and
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

import matplotlib
matplotlib.use('Agg')
import ipykernel

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from mtj_tracking.train.callback import PlotSegmentationCallback, PlotCallback
from mtj_tracking.train.model import AttentionVGG, normed_squared_distance, normed_rms_distance

tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)

from mtj_tracking.train.data import DataGenerator
import numpy as np
import os

base_path = os.path.join(os.environ.get("HOME"), 'final_run_v3', "mtj_training_att3_lr01_pc")
path = '/media/cleitner/Data/002_IEEE_EMBC/02_TrainingSet/Esaote'
labels_path = '/media/cleitner/Data/002_IEEE_EMBC/02_TrainingSet/20200121_combined.csv'

os.makedirs(os.path.join(base_path, 'weights'), exist_ok=True)

model = AttentionVGG(att='att3', height=64, width=128, channels=1, compatibilityfunction='pc',
                     optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0000001), loss=normed_squared_distance)
plot_model(model.model, os.path.join(base_path, 'mtj_model.png'), show_shapes=True)
model.model.summary()

start_epoch = model.loadWeights(base_path)

data_generator = DataGenerator(path, labels_path=labels_path)

x_train = np.array([d[1] for d in data_generator.data])
y_train = np.array([d[2] for d in data_generator.data])

x_valid = np.array([d[1] for d in data_generator.valid_data])
y_valid = np.array([d[2] for d in data_generator.valid_data])

train_model = model.model
scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.5 if (1 + epoch) % 25 == 0 else lr, verbose=1)

tboardcb = TensorBoard(log_dir=os.path.join(base_path, './logs'), histogram_freq=0, batch_size=3, write_graph=True,
                       write_grads=False,
                       write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                       embeddings_metadata=None)
checkpoint = ModelCheckpoint(os.path.join(base_path, 'weights', '{epoch}.hdf5'), save_weights_only=False)

plot_seg = PlotSegmentationCallback(x_valid, y_valid, model.model,
                                    model.attention_map_model, base_path=os.path.join(base_path, 'progress'))
plot = PlotCallback(x_valid, y_valid, model.model,
                    model.attention_map_model, base_path=os.path.join(base_path, 'progress'))
callbackslist = [scheduler, checkpoint, tboardcb, plot, plot_seg]
train_model.fit(x_train, y_train, 16, 300, callbacks=callbackslist, initial_epoch=start_epoch, shuffle=True, validation_data=(x_valid, y_valid))

train_model.save(os.path.join(base_path, 'trained_model.h5'))
