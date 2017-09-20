from __future__ import absolute_import, division, print_function, unicode_literals

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from xarray_filters import MLDataset
from xarray_filters.constants import FEATURES_LAYER
from xarray_filters.ts_grid_tools import (TSProbs, TSDescribe,
                                          resize_each_1d_slice,)
from xarray_filters.tests.test_data import ts_clustering_example
from xarray_filters.pipe_utils import for_each_array


def test_resize_each_1d_slice():
    arr = tuple(ts_clustering_example().data_vars.values())[0]
    rows = arr.shape[0]
    def func(arr):
        return (arr + 1)[:int(arr.shape[0] / 2)]
    dset = resize_each_1d_slice(arr, func,
                                axis=0, dim=None,
                                keep_attrs=True,
                                names=None,
                                chunks=None)
    assert isinstance(dset, MLDataset)


def test_ts_probs():
    s = TSProbs()
    num_bins = 152
    bins = np.linspace(-0.5 * num_bins, 0.5 * num_bins, num_bins + 1)
    s.set_params(layer='layer_1', bins=bins, log_probs=True)
    orig = ts_clustering_example()
    dset= s.transform(orig)
    assert FEATURES_LAYER in dset.data_vars
    assert dset.features.values.shape[1] == num_bins
    s.set_params(layer='layer_1', bins=bins, log_probs=False)
    dset2 = s.transform(orig)
    assert FEATURES_LAYER in dset2.data_vars
    assert dset2.features.values.shape[1] == num_bins
    s.set_params(layer='layer_1', bins=bins, log_probs=True)
    dset3 = s.transform(orig)
    assert FEATURES_LAYER in dset3.data_vars
    assert dset3.features.values.shape[1] == num_bins
    s.set_params(layer='layer_1', bins=152, log_probs=False)
    dset4 = s.transform(orig)
    assert FEATURES_LAYER in dset4.data_vars
    assert dset4.features.values.shape[1] == num_bins


def test_ts_describe():
    s = TSDescribe()
    s.set_params(layer='layer_1', axis=0)
    orig = ts_clustering_example()
    dset = s.transform(orig)
    layers = tuple(dset.layer)
    assert layers == ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')


def test_ts_describe_subset():
    s = TSDescribe()
    s.set_params(layer='layer_1', axis=0)
    orig = ts_clustering_example()
    dset = s.transform(orig)
    layers = tuple(dset.layer)
