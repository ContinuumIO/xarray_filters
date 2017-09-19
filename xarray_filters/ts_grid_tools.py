from __future__ import absolute_import, division, print_function, unicode_literals

'''
---------------------------------
``xarray_filters.ts_grid_tools``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
from collections import OrderedDict
import copy
from functools import partial
from itertools import product
import logging

import dask.array as da
import numpy as np
import pandas as pd
from scipy.stats import describe as scipy_describe
import xarray as xr

from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
from xarray_filters.pipe_utils import for_each_array
from xarray_filters.pipeline import Step
from xarray_filters.reshape import concat_ml_features

logger = logging.getLogger(__name__)
slc = slice(None)

def _ij_for_axis(axis, i, j):
    if axis == 0:
        return (slc, i, j)
    elif axis == 1:
        return (i, slc, j)
    elif axis == 2:
        return (i, j, slc)
    else:
        raise ValueError("Expected axis in (0, 1, 2)")

def _arr_1d_to_1d_reduce(values, axis, i, j):
    indices = _ij_for_axis(axis, i, j)
    values = values.__getitem__(*indices)
    arr_1d = func(values)
    return arr_1d


def resize_each_1d_slice(arr, func, axis=0, dim=None, keep_attrs=True, names=None):
    if axis is None and dim is not None:
        axis = arr.dims.index(dim)
    elif dim is None:
        dim = arr.dims[axis]
    else:
        dim = arr.dims[-1]
        axis = len(arr.dims) - 1
    dims = tuple(d for d in arr.values.dims if d != dim)
    shape = tuple(s for idx, s in enumerate(arr.values.shape)
                  if idx != axis)
    num_rows = np.prod(shape)
    new_arr = None
    for row, (i, j) in enumerate(product(*(range(s) for s in shape))):
        arr_1d = _arr_1d_to_1d_reduce(arr.values, axis, i, j)
        if new_arr is None:
            new_arr = da.empty((num_rows, arr_1d.size))
        new_arr[row, :] = arr_1d
    if keep_attrs:
        attrs = copy.deepcopy(arr.attrs)
    else:
        attrs = OrderedDict()
    np_arrs = tuple(getattr(arr, dim).values for dim in dims)
    index = pd.MultiIndex.from_product(np_arrs, names=dims)
    if names is None:
        names = np.arange(arr_1d.size)
    new_arr = xr.DataArray(new_arr,
                           coords=[(FEATURES_LAYER_DIMS[0], index),
                                  (FEATURES_LAYER_DIMS[1], names)],
                           dims=FEATURES_LAYER_DIMS,
                           attrs=attrs)
    new_dset = MLDataset({FEATURES_LAYER: new_arr}, attrs=attrs)
    return new_dset


def _describe(idxes, values):
    d = scipy_describe(values)
    t = (d.variance, d.skewness, d.kurtosis, d.minmax[0], d.minmax[1])
    median = np.median(values)
    std = np.std(values)
    non_param_skew = (d.mean - median) / std
    r = t + (median, std, non_param_skew)
    return np.array(r)[idxes]


def ts_describe(dset, axis=0, dim=None, layer=None, names=None, keep_attrs=True):
    '''scipy.describe on the `band` from kwargs
    that is a 3-D DataArray in X
    Parameters
    ----------

        X:  MLDataset or xarray.Dataset
        y:  passed through
        axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
        layer: The name of the DataArray in MLDataset to run scipy.describe on
        keep_attrs: TODO should default be True or False - docstring here ---

    Returns
    -------
        X:  MLDataset with DataArray class "features"
    '''
    if axis is None:
        axis = dims.index(dim)
    if layer is None:
        layers = tuple(dset.data_vars)
    def each_arr(arr, layer):
        arr = getattr(X, layer)
        default_names = ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')
        if names is None:
            names = default_names
        names = list(names)
        if len(set(default_names) & set(names)) == len(names):
            raise ValueError('Found names not in {}'.format(default_names))
        idxes = [default_names.index(name) for name in default_names]
        return resize_each_1d_slice(arr, partial(_describe, idxes),
                                    axis=axis, dim=dim, keep_attrs=keep_attrs,
                                    names=names)
    if isinstance(dset, xr.DataArray):
        layer = getattr(xr, 'name', None) or 'layer'
        return each_arr(dset[layer], layer)
    return concat_ml_features(*(each_arr(arr, layer)
                              for layer, arr in dset.data_vars.items()))


def _hist_1d(values, bins=None, log_counts=False, log_probs=False):
    hist, edges = np.histogram(values, bins)
    if log_counts:
        # add one half observation to avoid log zero
        small = 0.5
        hist[hist == 0] = small
        hist = np.log10(hist)
    else:
        small = 1.
        hist += small / hist.size
        extra = 1.0
    hist /= hist.sum()
    if log_probs:
        hist = np.log10(hist)
    return hist


def ts_probs(dset, bins=None, axis=0, dim=None, layer=None,
             log_counts=False, log_probs=False, names=None, keep_attrs=True):
    '''Fixed or unevenly spaced histogram binning for
    the time dimension of a 3-D cube DataArray in X
    Parameters:
        dset: MLDataset
        axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
        layer: The name of the DataArray in MLDataset to run scipy.describe on
        bins: Passed to np.histogram
        log_probs: Return probabilities associated with log counts? True / False
    '''
    if axis is None:
        axis = dims.index(dim)
    if layer is None:
        layer = tuple(dset.data_vars)
    def each_arr(arr, layer):
        arr = getattr(X, layer)
        return _hist_1d(values, bins=bins,
                        log_counts=log_counts,
                        log_probs=log_probs)
    if isinstance(dset, xr.DataArray):
        return each_arr(arr, layer)
    return concat_ml_features(*(each(arr, layer)
                              for layer, arr in dset.data_vars.items()))



class TSProbs(Step):
    def transform(self, *args, **kwargs):
        # TODO docstring from ts_probs
        kwargs['func'] = kwargs.get('func', self._func)
        super(TSProbs, self).__init__(**kwargs)
    fit = fit_transform = transform


class TSDescribe(Step):
    def transform(self, *args, **kwargs):
        # TODO docstring from ts_describe
        kwargs['func'] = kwargs.get('func', self._func)
        super(TSDescribe, self).__init__(**kwargs)
    fit = fit_transform = transform

__all__ = ['TSDescribe', 'TSProbs']