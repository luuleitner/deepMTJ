"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers

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


import tensorflow as tf
tf_config=tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=tf_config)
from keras.optimizers import SGD

import numpy as np
import os

from mtj_tracking.train.model import AttentionVGG, normed_squared_distance
from mtj_tracking.predict.dataIN import DataHandler
from mtj_tracking.predict.dataOUT import WriteVideoOut, csvPredictions, ATTWriteVideoOut


#########################################################################################################################
#                                               ADAPT File Path HERE
#########################################################################################################################
##
base = 'C:/Add Path here/deepMTJ/'     #------> Base to your Project Directory
Wpath = os.path.join(base, 'TrainedModels/', 'VGG-Attention-3.hdf5')    #------> Path to #deepMTJ Trained-Model (hdf5-File)
VDpath_in = os.path.join(base, 'Videos/')   #------> Path to Videos you want to predict (Relative to BASE Path!)
VDpath_out = os.path.join(base, 'Results/')  #------> Path where you want to save Result-Videos (Relative to BASE Path!)
##
#########################################################################################################################
#########################################################################################################################


# LOAD DATA
if not os.listdir(VDpath_in):
    print('NO VIDEO AVAILABLE FOR PREDICTION')
    exit()

predict_data_generator = DataHandler(VDpath_in, full_res=False)
x_test = np.array([d[1] for d in predict_data_generator.data])
Vlist = np.unique(np.array([d[0] for d in predict_data_generator.data]), return_index=True, return_inverse=False, return_counts=True, axis=None)

# LOAD MODEL BACKBONE
VGGA3model = AttentionVGG(att='att3', height=64, width=128, channels=1, compatibilityfunction='dp',
                     optimizer=SGD(lr=0.1, momentum=0.9, decay=0.0000001), loss=normed_squared_distance)

# LOAD TRAINED WEIGHTS
VGGA3model.model.load_weights(Wpath)

# PREDICT
predictions = VGGA3model.model.predict(x_test)
attention_maps = VGGA3model.attention_map_model.predict(x_test)

# WRITE VIDEO FILES
WriteVideoOut(x_test, predictions, VDpath_out, Vlist) #Predictions
#ATTWriteVideoOut(x_test, predictions, attention_maps, VDpath_out, Vlist) #Predictions with Attention Maps (only for display)

# WRITE CSV PREDICTION FILES
csvPredictions(x_test, predictions, VDpath_out, Vlist)
