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

import numpy as np
import pandas as pd
from scipy.stats import zscore


def pre_filter(signal, n_sigmas=2):
    s = signal.values
    z_scores = zscore(s, axis=0)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < n_sigmas)
    s[~filtered_entries[:, 0]] = np.nan
    s = pd.DataFrame(s,columns=['x','y'])
    s = s.interpolate(method='linear', axis=0, limit=20).ffill().bfill()
    return s

#
# Hampel filter adapted from https://github.com/erykml
#
def hampel_filter(input_series, window_size, n_sigmas=3):
    input_series = input_series.to_numpy()
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826
    indices = []
    for i in range((window_size), (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    return new_series