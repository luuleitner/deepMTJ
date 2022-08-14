import os
import glob
import numpy as np


def loadFiles(data_path):
    files = np.unique(glob.glob(os.path.join(data_path, '**', '*.avi'), recursive=True) + \
                      glob.glob(os.path.join(data_path, '**', '*.AVI'), recursive=True))
    return files.tolist()