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

from xarray_filters.constants import (FEATURES_LAYER_DIMS,
                                      FEATURES_LAYER,
                                      DASK_CHUNK_N)
from xarray_filters.pipe_utils import for_each_array
from xarray_filters.pipeline import Step
from xarray_filters.reshape import concat_ml_features

logger = logging.getLogger(__name__)
slc = slice(None)

def _validate_dims_axis(axis, dim, dims, shape):
    if axis is None and dim is not None:
        axis = dims.index(dim)
    elif dim is None and axis is not None:
        dim = dims[axis]
    else:
        dim = dims[-1]
        axis = len(dims) - 1
    return axis, dim


def _validate_layer(dset, layer):
    if layer is None:
        layer = tuple(dset.data_vars)
    elif isinstance(layer, str):
        layer = (layer,)
    return layer


def _arr_nd_to_1d(func, axis, values, i, j, **kw):
    indices = [i, j]
    indices.insert(axis, slc)
    values = values.__getitem__(indices)
    arr_1d = np.array(func(values, **kw))
    if arr_1d.ndim == 2 and arr_1d.shape[-1] in (0, 1):
        # TODO - see https://github.com/ContinuumIO/elm/issues/191
        # not sure if scikit-learn estimators all take 1-D y
        # data when column dim is singleton
        arr_1d = arr_1d.ravel()
    return arr_1d


def guess_chunks(shape):
    # TODO see other TODO 's regarding dask.array in place of numpy.array in
    # this module
    if len(shape) == 2:
        c = 1
        r =  DASK_CHUNK_N / shape[-1]
        return (r, c)
    else:
        raise NotImplementedError('guessing of chunks? or required? TODO')


def resize_each_1d_slice(arr, func, axis=0, dim=None, keep_attrs=True, names=None,
                         chunks=None, **kw):
    from xarray_filters.mldataset import MLDataset
    axis, dim = _validate_dims_axis(axis, dim, arr.dims, arr.shape)
    dims = tuple(d for d in arr.dims if d != dim)
    shape = tuple(s for idx, s in enumerate(arr.shape)
                  if idx != axis)
    print('dds', arr.dims, dims, shape)
    if chunks is None:
        chunks = guess_chunks(shape)  # TODO fix if implementing da.empty below
    num_rows = np.prod(shape)
    print('nr', num_rows)
    new_arr = None
    for row, (i, j) in enumerate(product(*(range(s) for s in shape))):
        arr_1d = _arr_nd_to_1d(func, axis, arr.values, i, j, **kw)
        if new_arr is None:
            new_arr = np.empty((num_rows, arr_1d.size)) # TODO da.empty here
        new_arr[row, :] = arr_1d
    if keep_attrs:
        attrs = copy.deepcopy(arr.attrs)
    else:
        attrs = OrderedDict()
    np_arrs = tuple(getattr(arr, dim).values for dim in dims)
    index = pd.MultiIndex.from_product(np_arrs, names=dims)
    if names is None:
        names = np.arange(arr_1d.size)
    coords = [(FEATURES_LAYER_DIMS[0], index),
              (FEATURES_LAYER_DIMS[1], np.array(names))]
    print('names', names, len(names))
    print('indx', [s.size for s in np_arrs])
    print('new', new_arr.shape)
    print('coords', coords, 'attrs', attrs, FEATURES_LAYER_DIMS)
    print('lennn', len(index.tolist()))
    new_arr = xr.DataArray(new_arr,
                           coords=coords,
                           dims=FEATURES_LAYER_DIMS,
                           attrs=attrs,
                           name=FEATURES_LAYER)
    new_dset = MLDataset(OrderedDict([(FEATURES_LAYER, new_arr)]),
                         attrs=attrs)
    print('new_dset', repr(new_dset))
    return new_dset


def _describe(idxes, values):
    d = scipy_describe(values)
    t = (d.variance, d.skewness, d.kurtosis, d.minmax[0], d.minmax[1])
    median = np.median(values)
    std = np.std(values)
    non_param_skew = (d.mean - median) / std
    r = t + (median, std, non_param_skew)
    return np.array(r)[idxes]


def ts_describe(dset, axis=0, dim=None, layer=None,
                names=None, keep_attrs=True, chunks=None):
    '''scipy.describe on the `band` from kwargs
    that is a 3-D DataArray in X
    Parameters
    ----------

        dset:  MLDataset or xarray.Dataset
        y:  passed through
        axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
        layer: The name of the DataArray in MLDataset to run scipy.describe on
        keep_attrs: TODO should default be True or False - docstring here ---
        chunks: TODO docstring

    Returns
    -------
        X:  MLDataset with DataArray class "features"
    '''
    layer = _validate_layer(dset, layer)
    default_names = ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')
    if names is None:
        names = default_names
    if len(set(default_names) & set(names)) != len(names):
        raise ValueError('Found names not in {}'.format(default_names))
    idxes = [default_names.index(name) for name in names]
    def each_arr(arr, layer, names, idxes):
        return resize_each_1d_slice(arr, partial(_describe, idxes),
                                    axis=axis, dim=dim, keep_attrs=keep_attrs,
                                    names=names, chunks=chunks)
    return concat_ml_features(*(each_arr(dset[layer], layer, names, idxes)
                                for layer in layer))


def _hist_1d(values, bins=None, log_counts=False, log_probs=False, **kw):
    hist, edges = np.histogram(values, bins)
    if log_counts:
        # add one half observation to avoid log zero
        small = 0.5
        hist[hist == 0] = small
        hist = np.log10(hist)
    else:
        small = 1.
        hist = hist + small / hist.size
    hist /= hist.sum()
    if log_probs:
        hist = np.log10(hist)
    return hist


def ts_probs(dset, bins=None, axis=0, dim=None, layer=None,
             log_counts=False, log_probs=False, names=None,
             keep_attrs=True, chunks=None):
    '''Fixed or unevenly spaced histogram binning for
    the time dimension of a 3-D cube DataArray in X
    Parameters:
        dset: MLDataset
        axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
        layer: The name of the DataArray in MLDataset to run scipy.describe on
        bins: Passed to np.histogram
        log_probs: Return probabilities associated with log counts? True / False
    '''
    layer = _validate_layer(dset, layer)
    def each_arr(arr, layer):
        return resize_each_1d_slice(arr, _hist_1d, bins=bins,
                        axis=axis, dim=dim, layer=layer,
                        log_counts=log_counts,
                        log_probs=log_probs,
                        names=names,
                        keep_attrs=keep_attrs,
                        chunks=chunks)
    return concat_ml_features(*(each_arr(dset[_], layer) for _ in layer))


class TSProbs(Step):
    bins = 50
    axis = 0
    dim = None
    layer = None
    log_counts = False
    log_probs = False
    names = None
    keep_attrs = True

    def transform(self, X, y=None):
        # TODO docstring from ts_probs - add into metaclass the docs copier
        # TODO y is ignored (document)
        # TODO - Default True / False on keep_attrs?
        params = self.get_params()
        print('params', params)
        return ts_probs(dset=X, **params)


class TSDescribe(Step):
    axis = 0
    dim = None
    layer = None
    names = None
    keep_attrs = True
    def transform(self, X, y=None):
        # TODO - Default True / False on keep_attrs?
        # TODO docstring from ts_describe
        # TODO y is ignored (document)
        params = self.get_params()
        print('params', params)
        return ts_describe(dset=X, **params)

__all__ = ['TSDescribe', 'TSProbs']
