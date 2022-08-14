from deepmtj.utils.filter import pre_filter, hampel_filter
import numpy as np
import pandas as pd

def postprocess(mtj_labels):

    XY_interp = []
    XY_fstage1 = []
    X_fstage2 = []
    Y_fstage2 = []

    for file in mtj_labels['file'].unique():

        # Interpolate lost data
        xy_interp = mtj_labels[['x', 'y']][mtj_labels['file'] == file].interpolate(method='linear',
                                                                                   axis=0,
                                                                                   limit=20).ffill().bfill().to_numpy()
        XY_interp.extend(xy_interp)

        # Apply prefilter (to reduce general noise)
        xy_fstage1 = pre_filter(xy_interp)
        XY_fstage1.extend(xy_fstage1)

        # # Apply Hampel filter (to reduce outliers)
        x_fstage2 = hampel_filter(xy_fstage1[:,0], 10, n_sigmas=2)
        X_fstage2.extend(x_fstage2)
        y_fstage2 = hampel_filter(xy_fstage1[:,1], 10, n_sigmas=2)
        Y_fstage2.extend(y_fstage2)

    columns = ['x interp', 'y interp', 'x fstage1', 'y fstage1', 'x fstage2', 'y fstage2']
    postprocess = pd.DataFrame(np.column_stack((XY_interp, XY_fstage1, X_fstage2, Y_fstage2)), columns=columns)
    mtj_labels = pd.concat([mtj_labels, postprocess], axis=1)

    return mtj_labels