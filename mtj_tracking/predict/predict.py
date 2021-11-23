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
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.measure import label

from mtj_tracking.data.generator import PredictionGenerator
from mtj_tracking.data.loader import Frame
from mtj_tracking.train.network import att_unet


def find_coordinates(prediction):
    detection = prediction >= 0.5
    l, num_features = label(detection, return_num=True)
    if num_features == 0:
        return (np.NAN, np.NAN)
    if num_features > 1:  # find the most likely detection (usually there is only 1 position)
        idx = np.argmax([(l == (i + 1)).sum() for i in range(num_features)]) + 1
    else:
        idx = 1
    detection_mask = (l == idx)
    y, x, _ = center_of_mass(prediction * detection_mask)
    return x, y


def track_videos(files, frame, frame_coordinates=True):
    model_path = os.path.join(str(Path.home()), '.deepMTJ', 'version1_0.tf')
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(str(Path.home()), '.deepMTJ'), exist_ok=True)
        urlretrieve('https://storage.googleapis.com/deepmtj/IEEEtbme_model_2021/2021_Unet_deepmtj_ieeetbme_model.tf', model_path, )

    model, _ = att_unet(128, 256, 1, 1)
    model.load_weights(model_path)

    data_generator = PredictionGenerator(files, frame=frame)
    predictions = model.predict_generator(data_generator, verbose=1)

    coordinates = np.array([find_coordinates(p) for p in predictions])

    x, y, w, h = frame.value
    result = {
        'file': [f for f, idx in data_generator.info],
        'frame_num': [idx for f, idx in data_generator.info],
        'x': coordinates[:, 0] if frame_coordinates else coordinates[:, 0] / 256 * w + x,
        'y': coordinates[:, 1] if frame_coordinates else coordinates[:, 1] / 128 * h + y}

    result_df = pd.DataFrame(result)
    return result_df


if __name__ == '__main__':
    result_df = track_videos([''], Frame.ESAOTE)
    result_df.to_csv('')
