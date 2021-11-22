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
import pickle
from multiprocessing import Pool
from random import sample
from typing import Dict

import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from skimage.transform import pyramid_expand

from mtj_tracking.train.model import distance, correct


class PlotCallback(Callback):

    def __init__(self, x, y, att_model, base_path='progress', prefix=None):
        super().__init__()
        idx = sample(range(0, len(x)), 8)

        self.data_x = x[idx]
        self.data_y = y[idx]
        self.att_model = att_model
        self.base_path = base_path
        self.prefix = prefix
        os.makedirs(base_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        n_samples = len(self.data_x)
        predictions, *att_maps = self.att_model.predict(self.data_x)
        n_columns = 3 + len(att_maps)

        fig, axs = plt.subplots(n_samples, n_columns, figsize=(n_columns * 6, n_samples * 3))
        for row, x, y, prediction, *att_map in zip(axs, self.data_x, self.data_y, predictions, *att_maps):
            row[0].imshow(x[..., 0], cmap='gray')
            row[0].axis('off')
            label = y
            for pos in np.argwhere(label[..., 0] == np.max(label[..., 0])):
                row[0].scatter(pos[1], pos[0], color='blue')
            label = prediction
            for pos in np.argwhere(label[..., 0] == np.max(label[..., 0])):
                row[0].scatter(pos[1], pos[0], color='magenta')

            # ground truth probability map
            row[1].imshow(x[..., 0], cmap='gray')
            row[1].axis('off')
            prob_map = np.copy(y[..., 0])
            prob_map[prob_map < 0.2] = np.NaN
            row[1].imshow(prob_map, vmin=0, vmax=1, alpha=0.5)
            # predicted probability map
            row[2].imshow(x[..., 0], cmap='gray')
            row[2].axis('off')
            prob_map = np.copy(prediction[..., 0])
            prob_map[prob_map < 0.2] = np.NaN
            row[2].imshow(prob_map, vmin=0, vmax=1, alpha=0.5)

            for ax, att in zip(row[3:], att_map):
                ax.axis('off')
                ax.imshow(att[..., 0], vmin=0, vmax=1)

        fig.tight_layout()
        plot_name = self.prefix + '_%03d.jpg' if self.prefix else '%03d.jpg'
        fig.savefig(os.path.join(self.base_path, plot_name) % epoch, dpi=100)
        plt.close('all')


class PlotSegmentationCallback(Callback):

    def __init__(self, x, y, model, attention_model, base_path='progress'):
        idx = sample(range(0, len(x)), 8)

        self.data_x = x[idx]
        self.data_y = y[idx]
        self.model = model
        self.attention_model = attention_model
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        n_samples = len(self.data_x)
        predictons = self.model.predict(self.data_x)
        attention_maps = self.attention_model.predict(self.data_x)
        n_columns = 1 + 2

        plt.figure(figsize=(n_columns * 6, n_samples * 3))
        for i, (x, y, predicton, *attention_map) in enumerate(
                zip(self.data_x, self.data_y, predictons, *attention_maps)):
            plt.subplot(n_samples, n_columns, i * n_columns + 1)
            plt.imshow(x[..., 0], cmap='gray')
            plt.axis('off')

            label = y
            for pos in np.argwhere(label[..., 0] >= 0.5):
                d_x = label[pos[0], pos[1], 1]
                d_y = label[pos[0], pos[1], 2]
                plt.scatter((pos[1] + d_x) * x.shape[0] / label.shape[0],
                            (pos[0] + d_y) * x.shape[1] / label.shape[1],
                            color='blue')
            label = predicton
            for pos in np.argwhere(label[..., 0] >= 0.5):
                d_x = label[pos[0], pos[1], 1]
                d_y = label[pos[0], pos[1], 2]
                plt.scatter((pos[1] + d_x) * x.shape[0] / label.shape[0],
                            (pos[0] + d_y) * x.shape[1] / label.shape[1],
                            color='red', alpha=label[pos[0], pos[1], 0])

            am2 = attention_map[1]
            am1 = pyramid_expand(attention_map[0], upscale=attention_map[1].shape[0] // attention_map[0].shape[0])
            am = np.sqrt(np.multiply(am1, am2))
            am = am[..., 0] if len(am.shape) == 3 else am

            plt.subplot(n_samples, n_columns, i * n_columns + 2)
            plt.imshow(x[..., 0], cmap='gray')
            scale = x.shape[0] // am.shape[0]
            att_map = pyramid_expand(am, upscale=scale) if scale > 1 else am
            plt.imshow(att_map, alpha=0.3, cmap='inferno')
            plt.axis('off')

            plt.subplot(n_samples, n_columns, i * n_columns + 3)
            plt.imshow(am, cmap='inferno')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.base_path, 'segmentation_%03d.jpg') % epoch, dpi=100)
        plt.close('all')


class ValidationCallback(Callback):

    def __init__(self, model, base_path, data_generators: Dict):
        self.model = model
        self.data_generators = data_generators
        self.base_path = base_path
        self.performance = {key: {'loss': [], 'accuracy': [], 'recall': [], 'distance': [], 'correct': []} for key in
                            self.data_generators.keys()}
        self.history_path = os.path.join(self.base_path, 'validation_history.pickle')

        if os.path.exists(self.history_path):
            with open(self.history_path, 'rb') as f:
                self.performance = pickle.load(f)

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        for key, gen in self.data_generators.items():
            with Pool(16) as p:
                data = p.map(gen.__getitem__, range(len(gen)))
            loss, accuracy, recall, _ = self.model.evaluate(np.concatenate([d[0] for d in data]),
                                                            np.concatenate([d[1] for d in data]))
            prediction = self.model.predict(np.concatenate([d[0] for d in data]))
            self.performance[key]['loss'] += [loss]
            self.performance[key]['accuracy'] += [accuracy]
            self.performance[key]['recall'] += [recall]
            self.performance[key]['distance'] += [distance(np.concatenate([d[1] for d in data]), prediction)]
            self.performance[key]['correct'] += [correct(np.concatenate([d[1] for d in data]), prediction)]
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.performance, f)
        self.plot()

    def plot(self):
        fig, axs = plt.subplots(5, 1, figsize=(8, 15))
        for key, values in self.performance.items():
            axs[0].plot(range(len(values['loss'])), values['loss'], '-o', label=key)
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].legend()
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        for key, values in self.performance.items():
            axs[1].plot(range(len(values['accuracy'])), values['accuracy'], '-o', label=key)
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].legend()
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        for key, values in self.performance.items():
            axs[2].plot(range(len(values['recall'])), values['recall'], '-o', label=key)
        axs[2].set_ylabel('Recall')
        axs[2].set_xlabel('Epoch')
        axs[2].legend()
        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        for key, values in self.performance.items():
            axs[3].plot(range(len(values['distance'])), values['distance'], '-o', label=key)
        axs[3].set_ylabel('Distance')
        axs[3].set_xlabel('Epoch')
        axs[3].legend()
        axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))

        for key, values in self.performance.items():
            axs[4].plot(range(len(values['correct'])), values['correct'], '-o', label=key)
        axs[4].set_ylabel('Correct')
        axs[4].set_xlabel('Epoch')
        axs[4].legend()
        axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(os.path.join(self.base_path, 'validation_history.jpg'), dpi=100)
        plt.close('all')
