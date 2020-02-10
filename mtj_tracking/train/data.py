import glob
import os
import random

import cv2
from pandas import read_csv
from skimage.transform import pyramid_reduce
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


class DataGenerator():

    def __init__(self, path, labels_path=None, valid_split=0.1, full_res = False, sample_size = False):
        self.full_res = full_res

        files = glob.glob(os.path.join(path, '**', '*.avi'), recursive=True) + \
                glob.glob(os.path.join(path, '**', '*.AVI'), recursive=True)

        labels_path = os.path.join(os.environ.get("HOME"), "labels.csv") if labels_path is None else labels_path
        df = read_csv(labels_path, header=None)
        labels = {d[0].lower().replace('.avi', ''): (d[1], d[2]) for idx, d in df.iterrows()}

        if sample_size:
            files = random.sample(files, sample_size)

        train_files, valid_files = train_test_split(files, test_size=valid_split, ) if valid_split > 0 else (files, [])

        self.data = self.loadVideos(train_files, labels)
        self.valid_data = self.loadVideos(valid_files, labels)

    def loadVideos(self, files, labels):
        data = []
        for file in tqdm(files):
            video_id, video = self.loadVideo(file)
            frames = [(self.createKey((video_id, i)),
                       self.adjustFrame(frame),
                       self.adjustLabels(labels[self.createKey((video_id, i))]))
                      for i, frame in enumerate(video) if self.createKey((video_id, i)) in labels.keys()]
            data = data + frames

        return data

    def loadVideo(self, path):
        vidcap = cv2.VideoCapture(path)

        video = []
        success, image = vidcap.read()
        while success:
            video.append(image)
            success, image = vidcap.read()
        video_id = os.path.basename(path).lower().replace('.avi', '')
        return video_id, video

    def createKey(self, active):
        return "%s_%05d" % (active[0], active[1])

    def adjustLabels(self, label):
        x = (label[0] - 210) / 4 / 128
        y = (label[1] - 130) / 4 / 64
        x = 1 if x > 1 else 0 if x < 0 else x
        y = 1 if y > 1 else 0 if y < 0 else y
        return x, y

    def adjustFrame(self, frame):
        frame = frame[130:130+256, 210:210+512]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 127.5 - 1
        full_res = frame
        frame = pyramid_reduce(frame, 4, order=3)
        frame = frame.reshape((*frame.shape, 1))
        if self.full_res:
            return frame, full_res
        return frame

class ImageDataGenerator(DataGenerator):

    def __init__(self, path, labels_path=None, valid_split=0.1, full_res = False, sample_size = False):
        self.full_res = full_res

        files = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True) + \
                glob.glob(os.path.join(path, '**', '*.JPG'), recursive=True)

        labels_path = os.path.join(os.environ.get("HOME"), "labels.csv") if labels_path is None else labels_path
        df = read_csv(labels_path, header=None)
        labels = {d[0].lower().replace('.avi', ''): (d[1], d[2]) for idx, d in df.iterrows()}

        if sample_size:
            files = random.sample(files, sample_size)



        data = self.loadImages(files, labels)

        train_videos, valid_videos = train_test_split(list(data.keys()), test_size=valid_split, ) if valid_split > 0 else (list(data.keys()), [])

        self.data = np.concatenate([data[video_id] for video_id in train_videos])
        self.valid_data = np.concatenate([data[video_id] for video_id in valid_videos]) if len(valid_videos) > 0 else []


    def loadImages(self, files, labels):
        data = {}
        for file in tqdm(files):
            video_id, frame_id, frame = self.loadImage(file)
            if frame_id in labels.keys():
                if video_id not in data:
                    data[video_id] = []
                frames = data[video_id]
                frame = self.adjustFrame(frame)
                label = self.adjustLabels(labels[frame_id])
                frames.append((frame_id, frame, label))

        return data

    def loadImage(self, path):
        video_id = os.path.basename(path).lower()[:-10]
        frame_id = os.path.basename(path).lower().replace('.jpg', '')
        frame = cv2.imread(path)
        return video_id, frame_id, frame