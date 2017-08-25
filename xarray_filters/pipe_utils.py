from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import xarray as xr

from xarray_filters.ml_features import MLDataset

__all__ = ['data_vars_kwargs', 'for_each_array']

def _attrs_and_type(dset, new_dset, return_dataset, keep_attrs, **kw):
    if return_dataset:
        new_dset = MLDataset(new_dset)
        if kw.get('keep_attrs', keep_attrs) or keep_attrs:
            if dset is not None:
                new_dset.attrs.update(dset.attrs)
    return new_dset


def data_vars_kwargs(return_dataset=True, keep_attrs=True):
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
                            data_vars_kwargs)
    '''
    def dec(func):
        def new_func(dset=None, **kw):
            new_dset = OrderedDict()
            kwargs = OrderedDict()
            if dset is not None:
                kwargs.update(dset.data_vars)
            kwargs.update(kw)
            new_dset = func(**kwargs)
            print(new_dset)
            new_dset = _attrs_and_type(dset, new_dset, return_dataset, keep_attrs, **kw)
            return new_dset
        return new_func
    return dec


def for_each_array(return_dataset=True):
    def dec(func):
        def new_func(**kw):
            new_dset = OrderedDict()
            for k, arr in kw.items():
                if isinstance(arr, xr.DataArray):
                    new_dset[k] = func(arr, **kw)
            if return_dataset:
                return MLDataset(new_dset)
            return new_dset
        return new_func
    return dec
