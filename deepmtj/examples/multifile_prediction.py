import sys
from os.path import abspath
from os.path import dirname as up
import os

path_to_lib = up(up(up(up(abspath(__file__)))))
sys.path.insert(0, path_to_lib)

from deepmtj.utils.auxilaryfunctions import loadFiles
from deepmtj.utils.postprocess import postprocess
from deepmtj.utils.plotting import plot_yFrame_shuffle
from deepmtj.predict.predict import track_videos
from deepmtj.data.loader import Frame


VIDEO_FILE_PATH = path_to_lib + '/deepMTJ/data'

# load avi-video files from directory
files = loadFiles(VIDEO_FILE_PATH)

# define crop size
crop_size = Frame.TELEMED

# track muscle-tendon junctions
mtj_labels_raw = track_videos(files, crop_size, export_frames=True)

# post-process labels
mtj_labels_filtered = postprocess(mtj_labels_raw)

# plot and save results
plot_yFrame_shuffle(mtj_labels_filtered, VIDEO_FILE_PATH, decimate=2)
mtj_labels_filtered.iloc[:, mtj_labels_filtered.columns != 'frame'].to_csv(os.path.join(VIDEO_FILE_PATH,
                                                                                        f'deep_mtj_results.csv'))
