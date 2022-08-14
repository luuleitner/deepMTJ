"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers

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

    Author: Christoph Leitner, Date: Aug. 2022
"""


import numpy as np
from os.path import join, splitext
from matplotlib import pyplot as plt

#########################################################################
# Plots Fits form a random shuffle or the complete set
def plot_yFrame_shuffle(mtj_labels, results_path, shuffle=None, decimate=None):

    for file in mtj_labels['file'].unique():
        if shuffle:
            selector = np.random.random_integers(0, len(mtj_labels['file'][mtj_labels['file'] == file])-1, shuffle)
            labels = mtj_labels[(mtj_labels['file'] == file) & (mtj_labels['frame_num'].isin(selector))]
            f_name = 'shuffle_plot'

            __rasterplot(labels,
                         results_path,
                         f_name,
                         title=file)

        else:
            if decimate:
                labels = mtj_labels[(mtj_labels['file'] == file) & (mtj_labels.index % decimate != 0)]
            sections = 3
            split_labels = np.array_split(labels, sections)

            for idx, labels in enumerate(split_labels):
                f_name = f'full_plot_{idx+1}-{sections}'
                __rasterplot(labels,
                             results_path,
                             f_name,
                             title=file)


#########################################################################
# Matplotlib object for: plot_yFrame_shuffle, plot_yFramesSD
def __rasterplot(A, path, f_name, title=None):
    n_columns = 8
    n_rows = int(np.ceil(len(A) / 8)) or 1

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 6, n_rows * 4))
    idx = 0
    for index, row in A.iterrows():
        axs.ravel()[idx].imshow(row['frame'])

        # x/y
        axs.ravel()[idx].scatter(float(row['x']), float(row['y']), c='b', marker='^', s=300, linewidth=2, label='raw')
        # x/y - interpolated
        axs.ravel()[idx].scatter(float(row['x interp']), float(row['y interp']), c='y', marker='o', s=300, linewidth=2, label='interp')
        # x/y - filter stage 1
        axs.ravel()[idx].scatter(float(row['x fstage1']), float(row['y fstage1']), c='g', marker='*', s=300, linewidth=2, label='f1')
        # x/y - filter stage 2
        axs.ravel()[idx].scatter(float(row['x fstage2']), float(row['y fstage2']), c='r', marker='+', s=60**2, linewidth=3, label='f2')
        axs.ravel()[idx].scatter(float(row['x fstage2']), float(row['y fstage2']), c='w', marker='o', s=30, linewidth=2)

        plt.axis('off')
        if title:
            axs.ravel()[idx].set_title(f'{splitext(title)[0]} - frame nbr: {index}', loc='center')
        else:
            axs.ravel()[idx].set_title(f'frame nbr: {index}', loc='center')
        idx = idx + 1

    legendEntries = ('raw', 'interpolated', 'filter stage 1', 'filter stage 2')
    fig.legend(legendEntries, ncol=len(legendEntries), loc="upper center")

    fig.tight_layout()
    fig.savefig(join(path, f'{splitext(title)[0]}_{f_name}.png'), dpi=100)
    plt.close('all')