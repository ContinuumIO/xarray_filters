from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from collections import OrderedDict
from functools import wraps

import xarray as xr
from xarray_filters.func_signatures import filter_args_kwargs


__all__ = ['data_vars_func',
           'for_each_array',
           'call_custom_func',]


def _keep_arrays(dset, new_dset):
    from xarray_filters.mldataset import MLDataset
    to_merge = OrderedDict()
    for layer, arr in dset.data_vars.items():
        if layer not in new_dset.data_vars:
            to_merge[layer] = arr
    return MLDataset(xr.merge((new_dset, MLDataset(to_merge))))


def _prepare_return_val(dset, new_dset, **kw):
    from xarray_filters.mldataset import MLDataset
    new_dset = MLDataset(new_dset)
    if kw.get('keep_arrays', False):
        new_dset = _keep_arrays(dset, new_dset)
    if kw.get('keep_attrs', False): # TODO convention on keep_attrs? True or False?
        if dset is not None:
            new_dset.attrs.update(dset.attrs)
    return new_dset


def data_vars_func(func):
    @wraps(func)
    def new_func(dset=None, *args, **kw):
        from xarray_filters.mldataset import MLDataset
        new_dset = OrderedDict()
        kwargs = OrderedDict()
        if dset is not None:
            dset = MLDataset(dset)
            kwargs.update(dset.data_vars)
        kwargs.update(kw)
        new_dset, args_kw = call_custom_func(func, *args, return_args_kw=True, **kwargs)
        kwargs.update(args_kw)
        new_dset = _prepare_return_val(dset, new_dset, **kwargs)
        return new_dset
    return new_func


def for_each_array(func):
    @wraps(func)
    def new_func(dset, *args, **kw):
        from xarray_filters.mldataset import MLDataset
        new_dset = OrderedDict()
        if hasattr(dset, 'data_vars'):
            items = dset.data_vars.items()
        elif hasattr(dset, 'items'):
            items = dset.items()
        else:
            raise ValueError('Expected a Dataset (or MLDataset) or dict as dset argument, but found argument of type {}'.format(type(dset)))
        for k, arr in items:
            new_dset[k] = call_custom_func(func, arr, *args, **kw)
        new_dset = _prepare_return_val(dset, new_dset, **kw)
        return new_dset
    return new_func


def call_custom_func(*args, **kwargs):
    assert len(args) > 0, 'xarray_filters.pipe_utils.call_custom_func requires at least one argument'
    func = args.pop()
    return_args_kw = kwargs.get('return_args_kw', False)
    if not callable(func):
        if args:
            args = list(args)
            arr = args.pop(0)
            func = getattr(arr, func)
        else:
            raise ValueError('TODO -improve message- expected a DataArray in *args')
    args_kw = filter_args_kwargs(func, *args, **kwargs)
    output = func(**args_kw)
    if return_args_kw:
        return output, args_kw
    return output

