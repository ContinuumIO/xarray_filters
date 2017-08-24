from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import xarray as xr

from xarray_filters.ml_features import MLDataset

__all__ = ['data_vars_iter']

def data_vars_iter(return_dataset=True,
                   array_method=None):
    def dec(func):
        def new_func(*args, **kw):
            dset = OrderedDict()
            kwargs = OrderedDict()
            if args and isinstance(args[0], (MLDataset, xr.Dataset)):
                kwargs.update(args[0].data_vars)
            kwargs.update(kw)
            for k, arr in kwargs.items():
                if not isinstance(arr, xr.DataArray):
                    continue
                if array_method:
                    method = getattr(arr, array_method)
                    dset[k] = func(*args, arr, **kwargs)
                else:
                    dset[k] = func(*args, arr, **kwargs)
            if return_dataset:
                return xr.Dataset(dset)
            return dset
        return new_func
    return dec
