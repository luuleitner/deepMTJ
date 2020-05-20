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


import glob
import os
import cv2
from skimage.transform import pyramid_reduce
from tqdm import tqdm


class DataHandler():

    def __init__(self, path, full_res=False):
        self.full_res = full_res

        files = glob.glob(os.path.join(path, '**', '*.avi'), recursive=True) + \
                glob.glob(os.path.join(path, '**', '*.AVI'), recursive=True)

        self.data = self.loadVideos(files)


    def loadVideos(self, files):
        data = []
        for file in tqdm(files):
            video_id, video = self.loadVideo(file)
            frames = [(self.createKey((video_id)),
                       self.adjustFrame(frame))
                      for i, frame in enumerate(video) if self.createKey((video_id))]
            data = data + frames
        return data

    def loadVideo(self, path):
        vidcap = cv2.VideoCapture(path)

        video = []
        success, image = vidcap.read()
        while success:
            video.append(image)
            success, image = vidcap.read()
        video_id = os.path.basename(path)
        return video_id, video

    def createKey(self, active):
        return "%s" % (active)

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