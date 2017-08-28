from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pandas as pd

__all__ = ['create_multi_index', 'multi_index_to_coords',]

def create_multi_index(arr):
    '''From DataArray arr make a pandas.MultiIndex for the arr.coords

    Parameters:
        :arr: xarray.DataArray

    Returns:
        :index: pandas.MultiIndex - The MultiIndex has index names
                taken from arr.dims and levels taken from arr.coords
    '''
    np_arrs = tuple(getattr(arr, dim).values for dim in arr.dims)
    index = pd.MultiIndex.from_product(np_arrs, names=arr.dims)
    return index


def multi_index_to_coords(arr, axis=0):
    '''Create an OrderedDict that will be a DataArray coords
    attribute, taking the dims and coord arrays from the DataArray
    arr's pandas.MultiIndex on one axis (axis=0 by default)

    Parameters:
        :arr: xarray.DataArray with a pandas.MultiIndex on one dimension
        :axis: Which axis has the pandas.MultiIndex (axis=0 by default)
    Returns:
        :coords: coordinates calculated from the MultiIndex
        :dims:   dimensions calculated from the MultiIndex
    '''
    dim = arr.dims[axis]
    multi = getattr(arr, dim)
    if not tuple(multi.coords.indexes) == (dim,):
        raise ValueError('MultiIndex has >1 dim ({}) - expected {}'.format())
    multi = multi.coords.indexes[dim]
    if not isinstance(multi, pd.MultiIndex):
        coords = OrderedDict([(dim, multi.values)])
        return coords, (dim,)
    if any(name is None for name in multi.names):
        raise ValueError('Expected MultiIndex with named components (found {})'.format(multi.names))
    np_arrs = (np.unique(x) for x in np.array(multi.tolist()).T)
    coords = OrderedDict(zip(multi.names, np_arrs))
    dims = tuple(coords)
    return coords, dims
