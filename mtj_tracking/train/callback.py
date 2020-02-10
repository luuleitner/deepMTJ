import os
from random import sample

from keras.callbacks import Callback
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import pyramid_expand


class PlotCallback(Callback):

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
        n_columns = 1 + 2 * len(attention_maps)

        plt.figure(figsize=(n_columns * 6, n_samples * 3))
        for i, (x, y, predicton, *attention_map) in enumerate(zip(self.data_x, self.data_y, predictons, *attention_maps)):
            plt.subplot(n_samples, n_columns, i * n_columns + 1)
            plt.imshow(x[..., 0], cmap='gray')
            plt.scatter(y[0] * x.shape[1], y[1] * x.shape[0], color='blue')
            plt.scatter(predicton[0] * x.shape[1], predicton[1] * x.shape[0], color='red')
            plt.axis('off')

            for j, am in enumerate(reversed(attention_map)):
                am = am[..., 0] if len(am.shape) == 3 else am

                plt.subplot(n_samples, n_columns, i * n_columns + 2 + j * 2)
                plt.imshow(x[..., 0], cmap='gray')
                scale = x.shape[0] // am.shape[0]
                att_map = pyramid_expand(am, upscale=scale) if scale > 1 else am
                plt.imshow(att_map, alpha=0.3, cmap='inferno')
                plt.axis('off')

                plt.subplot(n_samples, n_columns, i * n_columns + 3 + j * 2)
                plt.imshow(am, cmap='inferno')
                plt.axis('off')



        plt.tight_layout()
        plt.savefig(os.path.join(self.base_path, '%d.jpg') % epoch, dpi=100)
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
        for i, (x, y, predicton, *attention_map) in enumerate(zip(self.data_x, self.data_y, predictons, *attention_maps)):
            plt.subplot(n_samples, n_columns, i * n_columns + 1)
            plt.imshow(x[..., 0], cmap='gray')
            plt.scatter(y[0] * x.shape[1], y[1] * x.shape[0], color='blue')
            plt.scatter(predicton[0] * x.shape[1], predicton[1] * x.shape[0], color='red')
            plt.axis('off')

            am2 = attention_map[1]
            am1 = pyramid_expand(attention_map[0], upscale=attention_map[1].shape[0] // attention_map[0].shape[0])
            am = np.sqrt(np.multiply(am1, am2))
            am = am[...,0] if len(am.shape) == 3 else am

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
        plt.savefig(os.path.join(self.base_path, 'segmentation_%d.jpg') % epoch, dpi=100)
        plt.close('all')