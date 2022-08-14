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

import os
import glob
import numpy as np


def loadFiles(data_path):
    files = np.unique(glob.glob(os.path.join(data_path, '**', '*.avi'), recursive=True) + \
                      glob.glob(os.path.join(data_path, '**', '*.AVI'), recursive=True))
    return files.tolist()