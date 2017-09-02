from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
from functools import wraps

import xarray as xr

from xarray_filters.reshape import (chain, flatten,
                                    concat_ml_features)
from xarray_filters.multi_index import multi_index_to_coords
from xarray_filters.constants import FEATURES_LAYER

__all__ = ['from_features',
           'to_features',
           'MLDataset',
           'merge',]

def from_features(arr, axis=0):
    '''
    From a 2-D xarray.DataArray with a pandas.MultiIndex on one axis,
    return a MLDataset (xr.Dataset)

    Parameters:
        arr: 2-D xr.DataArray with
        axis: Axis with pandas.MultiIndex (default=0)
    Returns:
        MLDataset instance (inherits from xarray.Dataset)
    '''
    if arr.ndim > 2:
        raise ValueError('Expected 2D input arr but found {}'.format(arr.shape))
    coords, dims = multi_index_to_coords(arr, axis=axis)
    simple_axis = 0 if axis == 1 else 1
    simple_dim = arr.dims[simple_axis]
    simple_np_arr = getattr(arr, simple_dim).values
    shp = tuple(coords[dim].size for dim in dims)
    dset = OrderedDict()
    for j in range(simple_np_arr.size):
        val = arr[:, j].values.reshape(shp)
        layer = simple_np_arr[j]
        dset[layer] = xr.DataArray(val, coords=coords, dims=dims)
    return MLDataset(dset)


def ml_features_astype(dset, astype=None):
    '''TODO - Gui modify this function as needed
    to be consistent with your similar logic from
    PR 1.  Consider whether we need extra keywords
    for different situations, e.g. to control the
    acceptable return values
    '''
    if astype is None:
        astype = MLDataset
    if astype == MLDataset:
        return dset
    if astype == xr.Dataset:
        return xr.Dataset(dset)
    data_arr = dset[features_layer]
    if astype == xr.DataArray or (isinstance(astype, str) and 'dataarray' in astype.lower()):
        return data_arr
    if astype == pd.DataFrame or (isinstance(astype, str) or 'dataframe' in astype.lower()):
        return data_arr.to_dataframe()
    if astype in (np.ndarray, numpy.array) and isinstance(astype, str) and 'numpy' in astype:
        return data_arr.values
    raise ValueError('Expected one of ...TODO notes here on usage')


def to_features(dset, layers=None, row_dim=None,
                col_dim=None, trans_dims=None,
                features_layer=None, keep_attrs=False,
                astype=None):
    '''
    From an xarray.Dataset or MLDataset instance return a
    2-D xarray.DataArray or numpy.array for input as a
    feature matrix in tools like scikit-learn, by
    calling .ravel() (flattening) each N-D DataArray in dset

    Parameters:
        :row_dim: Name of the row dimension created by flattening the
                 coordinate arrays of each xarray.DataArray in dset.
                 The row dimension has a pandas.MultiIndex, e.g. if
                 xarray.DataArrays in dset have dims ('x', 'y')
                 then the pandas.MultiIndex has index names of ('x', 'y')
        :trans_dims: transpose
                 (becomes a pandas.MultiIndex)
        :col_dim: becomes column dimension
        :features_layer: Name of layer of returned MLDataset instance
        # TODO: Gui - make astype consistent with PR 2 (see comment at top of module)
        :astype: MLDataset instance by default or one of:
                'DataFrame' (returns a pandas.DataFrame)
                'numpy'
                'DataArray'
                'Dataset'

    It is assumed each xarray.DataArray in dset has the same coords/dims.

    Returns:
        MLDataset instance (inherits from xarray.Dataset)
    '''
    dset = flatten(dset, layers=layers, row_dim=row_dim,
                   col_dim=col_dim, trans_dims=trans_dims,
                   features_layer=features_layer,
                   keep_attrs=keep_attrs)
    return ml_features_astype(dset, astype=astype)


class MLDataset(xr.Dataset):
    '''Wraps xarray.Dataset for chainable preprocessors and
    reshaping to feature matricies that may be inputs to
    scikit-learn or similar models

    TODO - Gui: doctest MLDataset where needed in this
    class defintion
    '''

    def to_features(self,*args, **kwargs):
        '''* TODO Gui - wrap docstring for to_features'''
        return flatten(self, *args, **kwargs)

    def from_features(self, features_layer=None):
        '''* TODO wrap docstring for from_features'''
        if features_layer is None:
            features_layer = FEATURES_LAYER
        if not features_layer in self.data_vars:
            raise ValueError('features_layer ({}) not in self.data_vars'.format(features_layer))
        data_arr = self[features_layer]
        dset = from_features(data_arr)
        return dset

    def chain(self, func_args_kwargs, layers=None):
        '''
        '''
        return chain(self, func_args_kwargs, layers=layers)

    def concat_ml_features(self, *dsets,
                           features_layer=FEATURES_LAYER,
                           concat_dim=None,
                           keep_attrs=False):
        '''TODO - wrap docstring of concat_ml_features'''
        dsets = (self,) + tuple(dsets)
        return concat_ml_features(*dsets,
                                  features_layer=features_layer,
                                  concat_dim=concat_dim,
                                  keep_attrs=keep_attrs)
@wraps(xr.merge)
def merge(*args, **kw):
    return MLDataset(xr.merge(*args, **kw))

