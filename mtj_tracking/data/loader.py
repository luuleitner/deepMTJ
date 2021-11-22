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
import pickle
import random
from enum import Enum
from multiprocessing.pool import Pool

import cv2
import numpy as np
from pandas import read_csv
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Frame(Enum):
    ESAOTE = (185, 128, 580, 290)
    TELEMED = (88, 120, 446, 223)
    TELEMED_2 = (140, 50, 720, 360)
    TELEMED_3 = (140, 50, 720, 360)
    AIXPLORER = (200, 261, 1000, 500)


class DataLoader():

    def __init__(self, image_size=[128, 256]):
        # ------> Define Image Conversion Parameters
        self.image_size = image_size

    def loadDataSet(self, *data_paths,
                    labels_path=None,
                    valid_split=0.1,
                    sample_size=False, ):
        # ------> Load Filenames from the paths
        files = [f for data_path in data_paths for f in self.loadFiles(data_path)]

        # ------> Load Labels
        labels_path = os.path.join(os.environ.get("HOME"), "labels.csv") if labels_path is None else labels_path
        labels = self.loadLabels(labels_path)

        # ------> Include Data Sampling
        if sample_size:
            files = random.sample(files, sample_size)

        # ------> Create Training and Validation Datasets and Return
        train_files, valid_files = train_test_split(files, test_size=valid_split, ) if valid_split > 0 else (files, [])
        print('')
        print('Load frames of %d videos included in TRAINING dataset:' % len(train_files))
        data = self.loadVideos(train_files, labels)
        print('')
        print('Load frames of %d videos included in VALIDATION dataset:' % len(valid_files))
        valid_data = self.loadVideos(valid_files, labels)
        print('')
        print('Building Dataset...Done')
        return data, valid_data

    def loadFiles(self, data_path):
        return glob.glob(os.path.join(data_path, '**', '*.avi'), recursive=True) + \
               glob.glob(os.path.join(data_path, '**', '*.AVI'), recursive=True)

    def loadLabels(self, labels_path):
        df = read_csv(labels_path, header=None)
        labels = {d[0]: (d[1], d[2]) for idx, d in df.iterrows()}
        return labels

    def loadVideos(self, files, labels):
        data = []
        iter_elements = [(f, labels) for f in files]
        with Pool(24) as p:
            for frames in tqdm(p.imap_unordered(self.convertFile, iter_elements), total=len(iter_elements)):
                data = data + frames
        return data

    def convertFile(self, params):
        file, labels = params
        video_id, video = self.loadVideo(file)
        if 'E1' in video_id:
            frame = Frame.ESAOTE
        elif 'Te1' in video_id:
            frame = Frame.TELEMED
        elif 'Te2' in video_id:
            frame = Frame.TELEMED_3
        elif 'Te3' in video_id:
            frame = Frame.TELEMED_3
        elif 'Aix1' in video_id:
            frame = Frame.AIXPLORER
        else:
            raise Exception('Invalid videoID encountered: %s (%s)' % (video_id, file))
        frames = [(self.createKey((video_id, i)),
                   adjustFrame(image, frame, self.image_size),
                   adjustLabels(labels[self.createKey((video_id, i))], frame), video_id)
                  for i, image in enumerate(video) if self.createKey((video_id, i)) in labels.keys()]
        return frames

    def loadVideo(self, data_path):
        vidcap = cv2.VideoCapture(data_path)

        video = []
        success, image = vidcap.read()
        while success:
            video.append(image)
            success, image = vidcap.read()
        video_id = os.path.basename(data_path)
        return video_id, video

    def createKey(self, active):
        return "%s_%05d" % (active[0], active[1])


def adjustLabels(label, frame):
    x, y, w, h = frame.value
    label_x = (label[0] - x) / (w - 1)
    label_y = (label[1] - y) / (h - 1)

    # adjust out of frame labels
    label_x = 1 if label_x > 1 else 0 if label_x < 0 else label_x
    label_y = 1 if label_y > 1 else 0 if label_y < 0 else label_y
    return label_x, label_y


def loadVideo(data_path):
    vidcap = cv2.VideoCapture(data_path)

    video = []
    success, image = vidcap.read()
    while success:
        video.append(image)
        success, image = vidcap.read()
    video_id = os.path.basename(data_path)
    return video_id, video


def adjustFrame(image, frame, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x, y, w, h = frame.value
    image = image[y:y + h, x:x + w]

    image = resize(image, image_size, order=3)

    return image.astype(np.float32)


if __name__ == '__main__':
    base_path = ''
    train_path = ''
    labels_path = ''

    dl = DataLoader([256, 512])
    train, valid = dl.loadDataSet(os.path.join(train_path, 'Telemed'), labels_path=labels_path)

    with open(os.path.join(base_path, 'telemed.pickle'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(base_path, 'telemed_valid.pickle'), 'wb') as f:
        pickle.dump(valid, f)

    train, valid = dl.loadDataSet(os.path.join(train_path, 'Esaote'), labels_path=labels_path)

    with open(os.path.join(base_path, 'esaote.pickle'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(base_path, 'esaote_valid.pickle'), 'wb') as f:
        pickle.dump(valid, f)

    train, valid = dl.loadDataSet(os.path.join(train_path, 'Aixplorer'), labels_path=labels_path)

    with open(os.path.join(base_path, 'aixplorer.pickle'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(base_path, 'aixplorer_valid.pickle'), 'wb') as f:
        pickle.dump(valid, f)
