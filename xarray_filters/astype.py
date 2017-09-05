'''TODO - modify this module/function as needed
    to be consistent with similar logic from
    PR 2.  Consider whether we need extra keywords
    for different situations, e.g. to control the
    acceptable return values
    '''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import xarray as xr
import numpy as np

__all__ = ['ml_features_astype',]

def ml_features_astype(dset, astype=None):
    from xarray_filters.mldataset import MLDataset
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

