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
from multiprocessing import Pool

import numpy as np
from tensorflow.keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
from scipy.stats import multivariate_normal
from skimage.transform import resize
from sklearn.utils import shuffle
from tqdm import tqdm

from mtj_tracking.data.loader import Frame, adjustFrame, loadVideo


class PredictionGenerator(Sequence):

    def __init__(self, video_files, image_size=[128, 256], frame=Frame.ESAOTE, batch_size=10):
        for f in video_files:
            assert os.path.exists(f), 'Invalid video path: %s does not exist' % f
        self.image_size = image_size
        self.frame = frame
        frames = self.loadVideos(video_files)
        self.data = np.array([frame for video_id, idx, frame in frames])
        self.info = [(video_id, idx) for video_id, idx, frame in frames]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        x_batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.expand_dims(x_batch, -1)
        x_batch = (x_batch - np.mean(x_batch, axis=(1, 2, 3), keepdims=True)) / (
                    np.std(x_batch, axis=(1, 2, 3), keepdims=True) + 1e-8)
        return np.array(x_batch, dtype=np.float32)

    def loadVideos(self, files):
        data = []
        with Pool(24) as p:
            for frames in tqdm(p.imap(self.convertFile, files), total=len(files), desc='Loading videos'):
                data = data + frames
        return data

    def convertFile(self, file):
        video_id, video = loadVideo(file)
        frames = [(video_id, i, adjustFrame(image, self.frame, self.image_size)) for i, image in enumerate(video)]
        return frames

    def createKey(self, active):
        return "%s_%05d" % (active[0], active[1])


class TrainGenerator(Sequence):

    def __init__(self, x_data, y_data, batch_size=10, augmentation=True, image_shape=None):
        self.x_data, self.y_data = shuffle(x_data, y_data)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_transform = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True,
                                                  zoom_range=0.3, width_shift_range=0.1, height_shift_range=0.1,
                                                  shear_range=0.2, fill_mode='reflect')

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, idx):
        x_batch, y_batch = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size], self.y_data[
                                                                                           idx * self.batch_size:(
                                                                                                                             idx + 1) * self.batch_size]
        batch = [self.transform(x, y) for x, y in zip(x_batch, y_batch)]
        return np.array([d[0] for d in batch]), np.array([d[1] for d in batch])

    def transform(self, image, label):
        image = np.expand_dims(image, -1)
        x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        pos = np.dstack((x, y))
        y0 = int(label[1] * (pos.shape[0] - 1))
        x0 = int(label[0] * (pos.shape[1] - 1))
        target = multivariate_normal.pdf(pos, [y0, x0], [100, 100])
        target = np.expand_dims(target, axis=-1)
        if self.augmentation:
            params = self.image_transform.get_random_transform(image.shape)
            image = self.image_transform.apply_transform(image, params)
            target = self.image_transform.apply_transform(target, params)

        image = self.prepImage(image)
        target = self.prepTarget(target)

        return image, target

    def prepTarget(self, target):
        if self.image_shape:
            target = resize(target, self.image_shape, order=3)
        target = (target - np.min(target)) / (np.max(target) - np.min(target))  # normalize
        return target

    def prepImage(self, image):
        if self.image_shape:
            image = resize(image, self.image_shape, order=3)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image

    def on_epoch_end(self):
        self.x_data, self.y_data = shuffle(self.x_data, self.y_data)
