from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
import xarray as xr

from xarray_filters import MLDataset
from xarray_filters.constants import FEATURES_LAYER
from xarray_filters.ts_grid_tools import TSProbs, TSDescribe

def make_3d():
    arr = np.random.uniform(0, 1, 100000).reshape(100, 10, 100)
    return MLDataset({'layer_1': xr.DataArray(arr,
                                coords=[('time', np.arange(100)),
                                    ('x', np.arange(10)),
                                    ('y',np.arange(100))],
                                dims=('time', 'x', 'y'))})


def test_ts_probs():
    s = TSProbs()
    s.set_params(layer='layer_1', bin_size=0.5, num_bins=152, log_probs=True)
    orig = make_3d()
    X, _, _ = s.fit_transform(orig)
    assert hasattr(X, FEATURES_LAYER)
    assert X.features.values.shape[1] == 152
    s.set_params(layer='layer_1', bin_size=0.5, num_bins=152,
                 log_probs=False)
    X2, _, _ = s.fit_transform(orig)
    assert hasattr(X2, FEATURES_LAYER)
    assert X2.features.values.shape[1] == 152
    s.set_params(layer='layer_1', bin_size=0.5, num_bins=152, log_probs=True)
    X3, _, _ = s.fit_transform(orig)
    assert hasattr(X3, FEATURES_LAYER)
    assert X3.features.values.shape[1] == 152
    with pytest.raises(ValueError):
        s = TSProbs()
        s.fit_transform(orig)
    s.set_params(layer='layer_1', num_bins=152, log_probs=False)
    X4, _, _ = s.fit_transform(orig)
    assert hasattr(X4, FEATURES_LAYER)
    assert X4.features.values.shape[1] == 152


def test_ts_describe():
    s = TSDescribe()
    s.set_params(layer='layer_1', axis=0)
    orig = make_3d()
    X = s.fit_transform(orig)
    layers = tuple(X.layer)
    assert layers == ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')

