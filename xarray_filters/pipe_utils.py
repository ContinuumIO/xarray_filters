from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import xarray as xr

from xarray_filters.ml_features import MLDataset

__all__ = ['data_vars_iter']

def data_vars_iter(return_dataset=True, pass_data_vars=True):
    '''Decorate a function with a DataArray arg and DataArray
    return value.

    Parameters:
        :return_dataset: TODO: Gui please convert return_dataset
                         to the usage of astype pattern we are
                         using elsewhere.  The idea is that the
                         simple usage may return a dict of DataArrays
                         (so we need an astype for type dict)

                         Or maybe astype should be a decorator optionally?
                            (for when user's write their own function, e.g.
                            in the tests/test_reshape.py of this repo, the
                            iqr_standard function would have 2 decorators -
                            one for a data structure type and the current one
                            data_vars_iter)
        :pass_data_vars: If True, keyword arguments are updated to include
                      the layer names and DataArrays of the first arg (Dataset )
    '''
    def dec(func):
        def new_func(*args, **kw):
            dset = OrderedDict()
            kwargs = OrderedDict()
            if pass_data_vars:
                if args:
                    if isinstance(args[0], (MLDataset, xr.Dataset)):
                        kwargs.update(args[0].data_vars)
            kwargs.update(kw)
            for k, arr in kwargs.items():
                if not isinstance(arr, xr.DataArray):
                    continue
                dset[k] = func(*args, arr, **kwargs)
            if return_dataset:
                return xr.Dataset(dset)
            return dset
        return new_func
    return dec
