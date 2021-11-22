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

import pickle

import keras
import matplotlib

from mtj_tracking.data.generator import TrainGenerator
from mtj_tracking.train.callback import ValidationCallback, PlotCallback

matplotlib.use('Agg')

import tensorflow as tf
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from mtj_tracking.train.model import AttentionUNet, model_loss, model_accuracy, model_recall, class_loss

import numpy as np
import os

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)

###################################################################
## User Input (Path)
base_path = ''
train_path = os.path.join(base_path, 'train_v10')
prediction_path = os.path.join(train_path, 'images')
weights_path = os.path.join(train_path, 'weights')
os.makedirs(weights_path, exist_ok=True)
os.makedirs(prediction_path, exist_ok=True)
log_path = os.path.join(train_path, 'logs')
batch_size = 16
image_shape = (128, 256)

###################################################################
## Load Model
model = AttentionUNet(height=image_shape[0],
                      width=image_shape[1],
                      channels=1,
                      optimizer=Adam(1e-4),  # SGD(lr=0.001, momentum=0.9, decay=0.0000001),
                      loss=model_loss,
                      metrics=[model_accuracy, model_recall,
                               class_loss])  # ------> Load model with defined hyperparameters

# Write model Summary to File and Console
model.model.summary()  # ------> Plot summary of Layers and Parameters in output window (python console)

###################################################################
## Load Weights #------> (necessary only when continuing training)
start_epoch = model.loadWeights(train_path)
# model.model.load_weights('/home/cleitner/final_run/mtj_training_att3/weights/104.hdf5')

###################################################################
## Define Model and Training
train_model = model.model
scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.5 if (1 + epoch) % 25 == 0 else lr,
                                  verbose=1)  # ReduceLROnPlateau(verbose=1, patience=5)#

###################################################################
## Define Callbacks, Plots and Checkpoints during Training

# tensorboard grafics
train_model._get_distribution_strategy = lambda: None  # workaround for tensorboard bug
tboardcb = TensorBoard(log_dir=log_path,
                       histogram_freq=0,
                       write_graph=True,
                       write_grads=False,
                       write_images=True,
                       update_freq="epoch",
                       profile_batch=2,
                       embeddings_freq=0,
                       embeddings_metadata=None)

# model checkpoint weights
checkpoint = ModelCheckpoint(os.path.join(weights_path, '{epoch}.hdf5'), save_weights_only=False, period=10)

###################################################################
## Load and Generate Data for Training
with open(os.path.join(base_path, 'telemed_valid.pickle'), 'rb') as f:
    data = pickle.load(f)
x_valid_telemed = np.array([d[1] for d in data])
y_valid_telemed = np.array([d[2] for d in data])
valid_telemed_generator = TrainGenerator(x_valid_telemed, y_valid_telemed, batch_size=8, image_shape=image_shape,
                                         augmentation=False)

with open(os.path.join(base_path, 'esaote_valid.pickle'), 'rb') as f:
    data = pickle.load(f)
x_valid_esaote = np.array([d[1] for d in data])
y_valid_esaote = np.array([d[2] for d in data])
valid_esaote_generator = TrainGenerator(x_valid_esaote, y_valid_esaote, batch_size=8, image_shape=image_shape,
                                        augmentation=False)

with open(os.path.join(base_path, 'aixplorer_valid.pickle'), 'rb') as f:
    data = pickle.load(f)
x_valid_aixplorer = np.array([d[1] for d in data])
y_valid_aixplorer = np.array([d[2] for d in data])
valid_aixplorer_generator = TrainGenerator(x_valid_aixplorer, y_valid_aixplorer, batch_size=8, image_shape=image_shape,
                                           augmentation=False)

# init plot callbacks
esaote_plot = PlotCallback(*valid_esaote_generator[0],
                           model.attention_map_model, base_path=prediction_path, prefix='esaote')
telemed_plot = PlotCallback(*valid_telemed_generator[0],
                            model.attention_map_model, base_path=prediction_path, prefix='telemed')
aixplorer_plot = PlotCallback(*valid_aixplorer_generator[0],
                              model.attention_map_model, base_path=prediction_path, prefix='aixplorer')

validation_callback = ValidationCallback(train_model, train_path,
                                         {'Telemed': valid_telemed_generator, 'Esaote': valid_esaote_generator,
                                          'Aixplorer': valid_aixplorer_generator})

# Collect callback list
# callbackslist = [scheduler, checkpoint, tboardcb, plot, plot_seg]
callbacks = [checkpoint, esaote_plot, telemed_plot, aixplorer_plot, validation_callback]
esaote_plot.on_epoch_end(0)
telemed_plot.on_epoch_end(0)
aixplorer_plot.on_epoch_end(0)

###################################################################
## Start Training
step_size = 500

#### Telemed ####
print('Train Telemed')
# load data set
with open(os.path.join(base_path, 'telemed.pickle'), 'rb') as f:
    data = pickle.load(f)
x_train_telemed = np.array([d[1] for d in data])
y_train_telemed = np.array([d[2] for d in data])
train_generator = TrainGenerator(x_train_telemed, y_train_telemed, batch_size=batch_size, image_shape=image_shape)
valid_generator = TrainGenerator(x_valid_telemed, y_valid_telemed, batch_size=batch_size, image_shape=image_shape)
# training
train_model.fit_generator(train_generator, validation_data=valid_generator, epochs=step_size,
                          callbacks=callbacks, shuffle=True, use_multiprocessing=True, workers=16,
                          initial_epoch=start_epoch)
keras.models.save_model(train_model, os.path.join(train_path, 'telemed_model.tf'))

#### Esaote + Telemed ####
print('Train Esaote + Telemed')
# load data set
with open(os.path.join(base_path, 'esaote.pickle'), 'rb') as f:
    data = pickle.load(f)
x_train_esaote = np.array([d[1] for d in data])
y_train_esaote = np.array([d[2] for d in data])
#
start_epoch = max(start_epoch, step_size)  # update init epoch for next data set
train_generator = TrainGenerator(np.concatenate([x_train_esaote, x_train_telemed]),
                                 np.concatenate([y_train_esaote, y_train_telemed]),
                                 batch_size=batch_size, image_shape=image_shape)
valid_generator = TrainGenerator(np.concatenate([x_valid_esaote, x_valid_telemed]),
                                 np.concatenate([y_valid_esaote, y_valid_telemed]),
                                 batch_size=batch_size, image_shape=image_shape)
generator = train_model.fit_generator(train_generator, validation_data=valid_generator, epochs=step_size * 2,
                                      callbacks=callbacks, shuffle=True, use_multiprocessing=True, workers=16,
                                      initial_epoch=start_epoch)
keras.models.save_model(train_model, os.path.join(train_path, 'esaote_telemed_model.tf'))

#### combined ####
print('Train Esaote + Telemed + Aixplorer')
# load data set
with open(os.path.join(base_path, 'aixplorer.pickle'), 'rb') as f:
    data = pickle.load(f)
x_train_aixplorer = np.array([d[1] for d in data])
y_train_aixplorer = np.array([d[2] for d in data])
#
start_epoch = max(start_epoch, step_size * 2)  # update init epoch for next data set
train_generator = TrainGenerator(np.concatenate([x_train_esaote, x_train_telemed, x_train_aixplorer]),
                                 np.concatenate([y_train_esaote, y_train_telemed, y_train_aixplorer]),
                                 batch_size=batch_size, image_shape=image_shape)
valid_generator = TrainGenerator(np.concatenate([x_valid_esaote, x_valid_telemed, x_valid_aixplorer]),
                                 np.concatenate([y_valid_esaote, y_valid_telemed, y_valid_aixplorer]),
                                 batch_size=batch_size, image_shape=image_shape)
train_model.fit_generator(train_generator, validation_data=valid_generator, epochs=step_size * 3,
                          callbacks=callbacks, shuffle=True, use_multiprocessing=True, workers=16,
                          initial_epoch=start_epoch)
keras.models.save_model(train_model, os.path.join(train_path, 'esaote_telemed_aixplorer_model.tf'))
