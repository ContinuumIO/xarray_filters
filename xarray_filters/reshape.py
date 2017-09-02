'''
-----------------------
``earthio.reshape``
~~~~~~~~~~~~~~~~~~~~~~~
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple, OrderedDict
from copy import deepcopy
import os
import warnings

import attr
import dask
import numpy as np
import pandas as pd
import scipy.interpolate as spi
import xarray as xr

from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
from xarray_filters.multi_index import create_multi_index
from xarray_filters.pipe_utils import for_each_array, call_custom_func

__all__ = ['has_features_layer',
           'transpose',
           'flatten',
           'concat_ml_features',]

def _transpose_arr(data_arr, new_dims):
    if not len(set(new_dims) & set(data_arr.dims)) == len(new_dims):
        raise ValueError('At least one of new_dims is not an existing dim (new_dims {}, existing {})'.format(new_dims, data_arr.dims))
    return data_arr.transpose(*new_dims)


def transpose(es, new_dims, layers=None):
    '''Transpose an MLDataset - elm.pipeline.steps.Transpose
    Parameters:
        :new_dims: passed to xarray.DataArray.transpose
        :layers: list of DataArrays layers to include
    Returns:
        :MLDataset transposed
    '''
    from xarray_filters.ml_features import MLDataset
    if isinstance(es, (xr.Dataset, MLDataset)):
        trans = OrderedDict()
        layers = layers or tuple(es.data_vars)
        for layer in layers:
            trans[layer] = _transpose_arr(getattr(es, layer), new_dims)
        return MLDataset(trans, attrs=es.attrs)
    return _transpose_arr(es, new_dims)


def has_features_layer(dset, raise_err=True, features_layer=FEATURES_LAYER):
    '''Check if an MLDataset has a DataArray called "features"
    with dimensions (space, layer)
    Parameters:
        :dset: an MLDataset
        :raise_err: raise or not
    Returns:
        :bool: ``True``, ``False`` or raises ``ValueError``
               if not flat (raise_err=True)
    '''
    arr = getattr(dset, features_layer, None)
    if arr is None or not hasattr(arr, 'dims') or tuple(arr.dims) !=  FEATURES_LAYER_DIMS:
        msg = 'Expected an MLDataset/Dataset with DataArray "{}" and dims {}'
        if raise_err:
            raise ValueError(msg.format(features_layer, FEATURES_LAYER_DIMS))
        else:
            return False
    return True


def flatten(dset, layers=None, row_dim=None,
            col_dim=None, trans_dims=None,
            features_layer=None, keep_attrs=False):
    '''
    TODO - what is convention with keep_attrs: default=True or False?
    '''
    from xarray_filters.ml_features import MLDataset
    arrs = []
    if features_layer is None:
        features_layer = FEATURES_LAYER
    if row_dim is None:
        row_dim = FEATURES_LAYER_DIMS[0]
    if col_dim is None:
        col_dim = FEATURES_LAYER_DIMS[1]
    if layers is None:
        layers = tuple(dset.data_vars)
    for layer in layers:
        if not layer in dset.data_vars:
            raise ValueError('TODO - message')
        arr = dset[layer]
        coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
        if trans_dims is not None:
            if tuple(trans_dims) != tuple(arr.dims):
                arr = transpose(arr, trans_dims)
                coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
                attrs = arr.attrs
        attrs = deepcopy(attrs)
        if len(dims) == 1:
            row_dim = dims[0]
            index = getattr(arr, row_dim)
        else:
            index = create_multi_index(arr)
        val = val.ravel()[:, np.newaxis]
        coords = OrderedDict([(row_dim, index),
                              (col_dim, [layer])])
        new_dims = (row_dim, col_dim)
        arr = xr.DataArray(val, coords=coords,
                           dims=new_dims, attrs=attrs)
        arrs.append(arr)
    dims, siz = _same_size_dims_arrs(*arrs)
    if not all(col_dim in arr.dims for arr in arrs):
        raise ValueError('TODO - document how one ends up here {}'.format(layers))
    new_arr = xr.concat(arrs, dim=col_dim)
    if not keep_attrs:
        attrs = OrderedDict()
    return MLDataset(OrderedDict([(features_layer, new_arr)]),attrs=attrs)


def _same_size_dims_arrs(*arrs, raise_err=True):
    '''Check if all DataArrays in arrs have same size and same dims

    Parameters:
        :raise_err: If True, raise ValueError if dims/sizes differ
                    else return True/False
    '''
    siz = None
    dims = None
    for arr in arrs:
        if siz is not None and siz != arr.size:
            if raise_err:
                raise ValueError('Expected arrays of same size but found {} and {}'.format(siz, arr.size))
            return False
        if dims is not None and tuple(arr.dims) != dims:
            if raise_err:
                raise ValueError('Expected arrays of same dims but found {} and {}'.format(dims, arr.dims))
            return False
    return dims, siz


def subset_layers(dset, layers):
    from xarray_filters.ml_features import MLDataset
    arrs = (dset[layer] for layer in layers
            if layer in dset.data_vars)
    new_dset = MLDataset(OrderedDict(zip(layers, arrs)))
    return new_dset


def format_chain_args(trans):
    '''TODO - Document the list of list structures
    that can be passed transforms.  See
    comments in tests/test_reshape.py for now

    Parameters:
        :trans: "transforms" arguments to MLDataset.chain

    Returns:
        list of lists like [[func1, args1, kwargs1],
                            [func2, args2, kwargs2]]
        to be passed to MLDataset.pipe(func, *args, **kwargs) on
        each func, args, kwargs.  Note
    '''
    output = []
    if callable(trans):
        output.append([trans, [], {}])
    elif isinstance(trans, (tuple, list)):
        trans = list(trans)
        for tran in trans:
            if callable(tran):
                output.append([tran, [], {}])
            elif isinstance(tran, (list, tuple)) and tran:
                tran = list(tran)
                if isinstance(tran[-1], dict):
                    kw = tran[-1]
                    kw_idx = len(tran) - 1
                else:
                    kw = dict()
                    kw_idx = None
                func = [idx for idx, _ in enumerate(tran)
                        if isinstance(_, str) or callable(_)]
                if not func:
                    raise ValueError('Expected a string DataArray method name or a callable in {}'.format(tran))
                args = [_ for idx, _ in enumerate(tran)
                        if idx != kw_idx and idx not in func]
                func = tran[func[0]]
                tran = [func, args, kw]
                output.append(tran)
    return output


def chain(dset, func_args_kwargs, layers=None):
    from xarray_filters.pipe_utils import for_each_array
    func_args_kwargs = format_chain_args(func_args_kwargs)
    if layers is not None:
        dset = subset_layers(dset, layers=layers)
    for func, args, kwargs in func_args_kwargs:
        if not callable(func):
            func = for_each_array(func)
        dset = dset.pipe(func, *args, **kwargs)
    return dset


def concat_ml_features(*dsets,
                       features_layer=FEATURES_LAYER,
                       concat_dim=None,
                       keep_attrs=False):

    '''Concatenate MLDataset / Dataset (dsets) along concat_dim
    (by default the column dimension, typically called "layer")

    Parameters:
        :dsets: Any number of MLDataset / Dataset objects that are
                2D
        :features_layer: Typically "layer", the column dimension
        :concat_dim: If None, the column dimension is guessed
        :keep_attrs: If True, keep the attrs of the first dset in *dsets
    TODO - Gui: This could use the astype logic discussed elsewhere?


    '''

    # TODO True or False (convention?)
    from xarray_filters.ml_features import MLDataset
    if not dsets:
        raise ValueError('No MLDataset / Dataset arguments passed.  Expected >= 1')
    if keep_attrs:
        attrs = deepcopy(dsets[0].attrs)
    else:
        attrs = OrderedDict()
    concat_dim = concat_dim or FEATURES_LAYER_DIMS[1]
    data_arrs = []
    for dset in dsets:
        if not isinstance(dset, (MLDataset, xr.Dataset)):
            raise ValueError('TODO -error message here')
        data_arr = dset.data_vars.get(features_layer, None)
        if data_arr is None:
            raise ValueError('TODO -error message here')
        data_arrs.append(data_arr)
    data_arr = xr.concat(data_arrs, dim=concat_dim)
    return MLDataset(OrderedDict([(features_layer, data_arr)]), attrs=attrs)

