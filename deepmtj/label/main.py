"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers

	Copyright (C) 2020 by the authors: Jarolim Robert (University of Graz), <robert.jarolim@uni-graz.at> and
	Leitner Christoph (Graz University of Technology), <christoph.leitner@tugraz.at>.
	
	Please note: main.ui was compiled in open-source Qt (protected by GPLv3)

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

import matplotlib
from pandas import read_csv
from sklearn.utils import shuffle

matplotlib.use('QT5Agg')

import sys

from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import cv2
import numpy as np

Ui_MainWindow, QtBaseClass = uic.loadUiType('main.ui')


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.folder_select_button.clicked.connect(self.selectFolder)
        self.start_button.clicked.connect(self.start)
        self.plot_widget.coordinate_selection.connect(self.labelFrame)
        self.backward_button.clicked.connect(self.loadPrevious)
        self.forward_button.clicked.connect(self.loadNext)
        self.set_button.clicked.connect(self.setFrame)
        self.action_change_label_path.triggered.connect(self.changeLabelPath)

        self.queue = []
        self.queue_idx = None
        self.videos = {}
        self.labels = {}
        self.labels_path = os.path.join(os.environ.get("HOME"), "labels.csv")
        self.loadLabels()

        self.max_videos = 5

    def loadVideo(self, path):
        self.video_input.setText(path)
        vidcap = cv2.VideoCapture(path)

        video = []
        success, image = vidcap.read()
        while success:
            video.append(image)
            success, image = vidcap.read()
        video_id = os.path.basename(path)
        self.videos[video_id] = video

    def loadNext(self):
        if len(self.queue) == 0:
            return
        self.queue_idx = self.queue_idx + 1 if self.queue_idx is not None else 0
        if self.queue_idx >= len(self.queue):
            if len(self.files) > 0:
                self.loadVideoBatch()
                self.queueVideos()
                self.loadNext()
                return
            self.queue_idx = len(self.queue) - 1
        active = self.queue[self.queue_idx]

        self.changeFrame(active)

    def loadPrevious(self):
        if len(self.queue) == 0:
            return
        self.queue_idx = self.queue_idx - 1 if self.queue_idx is not None else -1
        if self.queue_idx < 0 and len(self.queue) == 0:
            self.queue_idx = None
            return
        if self.queue_idx < 0:
            self.queue_idx = 0
        active = self.queue[self.queue_idx]

        self.changeFrame(active)

    def changeFrame(self, active):
        video = self.videos[active[0]]
        self.frame_spin.setMaximum(len(video) - 1)
        self.frame_slider.setMaximum(len(video) - 1)
        self.frame_spin.setValue(active[1])
        self.frame_slider.setValue(active[1])
        self.plot_widget.loadFrame(video[active[1]], self.labels.get(self.createKey(active), None))
        self.video_input.setText(active[0])

    def selectFolder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(directory=self.folder_input.text(), options=options)
        if path:
            self.folder_input.setText(path)

    def start(self):
        self.plot_widget.clear()
        files = glob.glob(os.path.join(self.folder_input.text(), '**', '*.avi'), recursive=True) + glob.glob(
            os.path.join(self.folder_input.text(), '**', '*.AVI'), recursive=True)
        self.files = files

        # queue maximum of videos
        self.loadVideoBatch()
        self.queueVideos()
        self.loadNext()

    def loadVideoBatch(self):
        self.videos = {}
        for i in range(self.max_videos):
            self.loadVideo(self.files.pop())

    def queueVideos(self):
        self.queue = []
        n_per_video = self.per_video_spin.value()
        for video_id, video in self.videos.items():
            if self.per_video_radio.isChecked():
                sample_step = int(np.ceil(len(video) / (n_per_video + 1)))
                self.queue = self.queue + [(video_id, i) for i in range(sample_step, len(video), sample_step)]
            else:
                self.queue = self.queue + [(video_id, i) for i in range(len(video))]
        if self.labeled_radio.isChecked():
            self.queue = [q for q in self.queue if self.createKey(q) in self.labels]
        if self.unlabeled_radio.isChecked():
            self.queue = [q for q in self.queue if self.createKey(q) not in self.labels]
        if self.random_radio.isChecked():
            self.queue = shuffle(self.queue)
        if self.sample_radio.isChecked():
            self.queue = self.queue[::self.sample_spin.value()]
        self.queue_idx = None

    def labelFrame(self, coordinates):
        active = self.queue[self.queue_idx]
        self.labels[self.createKey(active)] = coordinates
        if os.path.exists(self.labels_path):
            os.remove(self.labels_path)
        with open(self.labels_path, 'w') as f:
            for key, value in self.labels.items():
                f.write("%s,%d,%d\n" % (key, value[0], value[1]))
        self.loadNext()

    def createKey(self, active):
        return "%s_%05d" % (active[0], active[1])

    def setFrame(self):
        if self.queue_idx is None:
            return
        active = self.queue[self.queue_idx]
        self.queue.insert(self.queue_idx + 1, (active[0], self.frame_spin.value()))

        self.loadNext()

    def changeLabelPath(self):
        file_path, _ = QFileDialog.getSaveFileName(directory=self.labels_path)
        if not file_path:
            return
        self.labels_path = file_path
        self.loadLabels()

    def loadLabels(self):
        if not os.path.exists(self.labels_path):
            self.labels = {}
        else:
            df = read_csv(self.labels_path, header=None)
            self.labels = {d[0]: (d[1], d[2]) for idx, d in df.iterrows()}


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
