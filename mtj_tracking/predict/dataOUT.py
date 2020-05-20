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


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from skimage.transform import pyramid_expand

import numpy as np
import os
import cv2
from tqdm import tqdm


def WriteVideoOut(x_data, predictions, Vdpath_out, Vlist):
        print('...Write Video Clips')
        for i in tqdm(range(len(Vlist[1]))):
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(Vdpath_out, ('deepMTJ_%s') % Vlist[0][i]), fourcc, 25.0, (640, 480))

            for j in range(Vlist[1][i], Vlist[1][i]+Vlist[2][i]):
                frame_fig = plt.figure()
                ax = frame_fig.gca()
                ax.margins(0)
                ax.axis('off')
                frame_fig.tight_layout(pad=0)

                plt.imshow(np.squeeze(x_data[j]), cmap='gray')
                plt.rcParams['lines.linewidth'] = 4
                plt.rcParams['lines.markersize'] = 30
                plt.scatter(predictions[j][0] * x_data.shape[2], predictions[j][1] * x_data.shape[1], color='red',
                            marker='+')
                plt.rcParams['lines.linewidth'] = 3
                plt.rcParams['lines.markersize'] = 5
                plt.scatter(predictions[j][0] * x_data.shape[2], predictions[j][1] * x_data.shape[1], color='white',
                            marker='o')
                plt.axis('off')

                frame_fig.canvas.draw()
                image_from_plot_RGB = np.frombuffer(frame_fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot_RGB.reshape(frame_fig.canvas.get_width_height()[::-1] + (3,))
                im_bgr = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
                out.write(im_bgr)

                plt.cla()
                plt.clf()
                plt.close(frame_fig)

            cap.release()
            out.release()
            cv2.destroyAllWindows()


def ATTWriteVideoOut(x_data, predictions, attention_map, Vdpath_out, Vlist):
    print('...Write Video Clips')
    for i in tqdm(range(len(Vlist[1]))):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(Vdpath_out, ('deepMTJ_%s') % Vlist[0][i]), fourcc, 25.0, (640, 480))

        for j in range(Vlist[1][i], Vlist[1][i] + Vlist[2][i]):
            frame_fig = plt.figure()
            ax = frame_fig.gca()
            ax.margins(0)
            ax.axis('off')
            frame_fig.tight_layout(pad=0)

            am2 = attention_map[1][j]
            am1 = pyramid_expand(attention_map[0][j], upscale=attention_map[1][j].shape[0] // attention_map[0][j].shape[0])
            am = np.sqrt(np.multiply(am1, am2))
            am = am[..., 0] if len(am.shape) == 3 else am

            plt.imshow(np.squeeze(x_data[j]), cmap='gray')
            plt.rcParams['lines.linewidth'] = 4
            plt.rcParams['lines.markersize'] = 30
            plt.scatter(predictions[j][0] * x_data.shape[2], predictions[j][1] * x_data.shape[1], color='white', marker='+')
            plt.rcParams['lines.linewidth'] = 3
            plt.rcParams['lines.markersize'] = 5
            plt.scatter(predictions[j][0] * x_data.shape[2], predictions[j][1] * x_data.shape[1], color='black', marker='o')

            scale = x_data.shape[1] // am.shape[0]
            att_map = pyramid_expand(am, upscale=scale) if scale > 1 else am

            #color_map = plt.cm.get_cmap('inferno')
            #reversed_cmap = color_map.reversed()

            plt.imshow(att_map, alpha=0.4, cmap='inferno')
            plt.axis('off')

            frame_fig.canvas.draw()
            image_from_plot_RGB = np.frombuffer(frame_fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot_RGB.reshape(frame_fig.canvas.get_width_height()[::-1] + (3,))
            im_bgr = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
            out.write(im_bgr)

            plt.cla()
            plt.clf()
            plt.close(frame_fig)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def csvPredictions(x_data, predictions, Vdpath_out, Vlist):
    print('...Write Prediction csv-File')
    px_width = x_data.shape[2]
    px_height = x_data.shape[1]
    px_WiHi = np.array([px_width,px_height])
    Pred_corr = np.multiply(predictions, px_WiHi)

    for i in tqdm(range(len(Vlist[1]))):
        Video_pred = np.empty(shape=[1,2])
        for j in range(Vlist[1][i], Vlist[1][i] + Vlist[2][i]):
            Frame_pred = Pred_corr[[j],:]
            Video_pred = np.append(Video_pred, Frame_pred, axis=0)

        csv_path = os.path.join(Vdpath_out, ('deepMTJ_%dX%d_%s.csv' % (px_height, px_width, Vlist[0][i])))
        np.savetxt(csv_path, np.squeeze(Video_pred), delimiter=",")