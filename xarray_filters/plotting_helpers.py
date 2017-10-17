from __future__ import absolute_import, division, print_function, unicode_literals

'''
------------------------------------

``earthio.filters.plotting_helpers``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
from collections import Sequence

import numpy as np

def plot_3d(X, layers, title='', scale=None, axis_labels=True,
            **imshow_kwargs):
    '''Plot a true or pseudo color image of 3 layers

    Parameters:
        :X: MLDataset or xarray.Dataset
        :layers: list of 3 layer names that are in X
        :title: title for figure
        :scale: divide all values by this (e.g. 2\*\* 16 for uint16)
        :axis_labels: True / False show axis_labels
        :\*\*imshow_kwargs: passed to imshow

    Returns:
        :(arr, fig): where arr is the 3-D numpy array and fig is the figure

    '''
    import matplotlib.pyplot as plt
    arr = None
    scale = 1 if scale is None else scale
    for idx, layer in enumerate(layers):
        val = getattr(X, layer).values
        if idx == 0:
            arr = np.empty((val.shape) + (len(layers),), dtype=np.float32)
        if isinstance(scale, Sequence):
            s = scale[idx]
        else:
            s = scale
        arr[:, :, idx] = val.astype(np.float64) / s
    plt.imshow(arr, **imshow_kwargs)
    plt.title('{:^100}'.format(title))
    fig = plt.gcf()
    if not axis_labels:
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
    return (arr, fig)
