from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
from functools import wraps

import xarray as xr
from xarray_filters.func_signatures import filter_args_kwargs


__all__ = ['data_vars_kwargs',
           'for_each_array',
           'call_custom_func',
           'return_dataset',]

def _attrs_and_type(dset, new_dset, return_dataset, **kw):
    from xarray_filters.ml_features import MLDataset
    if return_dataset:
        new_dset = MLDataset(new_dset)
        if kw.get('keep_attrs'):
            if dset is not None:
                new_dset.attrs.update(dset.attrs)
    return new_dset


def data_vars_kwargs(func):
    @wraps(func)
    def new_func(dset=None, **kw):
        new_dset = OrderedDict()
        kwargs = OrderedDict()
        if dset is not None:
            kwargs.update(dset.data_vars)
        kwargs.update(kw)
        new_dset = call_custom_func(func, **kwargs)
        new_dset = _attrs_and_type(dset, new_dset, return_dataset, **kw)
        return new_dset
    return new_func


def for_each_array(func):
    @wraps(func)
    def new_func(dset=None, *args, **kw):
        from xarray_filters.ml_features import MLDataset
        new_dset = OrderedDict()
        items = kw.items()
        if dset is not None:
            items = dset.data_vars.items()
        for k, arr in items:
            if isinstance(arr, xr.DataArray):
                new_dset[k] = call_custom_func(func, arr, **kw)
        if return_dataset:
            return MLDataset(new_dset)
        return new_dset
    return new_func


def return_dataset(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        from xarray_filters.ml_features import MLDataset
        return MLDataset(func(*args, **kwargs))
    return new_func


def call_custom_func(func, *args, layers_list=None, **kwargs):
    from xarray_filters.ml_features import MLDataset
    args_kw, missing = filter_args_kwargs(func, *args, **kwargs)
    if missing == 0:
        return func(**args_kw)
    else:
        raise NotImplmentedError('Either a Dataset or DataArray is a missing argument?')