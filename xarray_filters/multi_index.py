from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pandas as pd

__all__ = ['create_multi_index', 'multi_index_to_coords',]

def create_multi_index(arr):
    np_arrs = tuple(getattr(arr, dim).values for dim in arr.dims)
    index = pd.MultiIndex.from_product(np_arrs, names=arr.dims)
    return index


def multi_index_to_coords(arr, axis=0):
    dim = arr.dims[axis]
    multi = getattr(arr, dim)
    if not tuple(multi.coords.indexes) == (dim,):
        raise ValueError('MultiIndex has >1 dim ({}) - expected {}'.format())
    multi = multi.coords.indexes[dim]
    if any(name is None for name in multi.names):
        raise ValueError('Expected MultiIndex with named components (found {})'.format(multi.names))
    np_arrs = (np.unique(x) for x in np.array(multi.tolist()).T)
    coords = OrderedDict(zip(multi.names, np_arrs))
    dims = tuple(coords)
    return coords, dims
