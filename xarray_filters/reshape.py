'''
-----------------------
``earthio.reshape``
~~~~~~~~~~~~~~~~~~~~~~~
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple, OrderedDict
import copy
import os

import attr
import dask
import numpy as np
import pandas as pd
import scipy.interpolate as spi
import xarray as xr

from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
from xarray_filters.multi_index import create_multi_index

__all__ = ['has_features_layer',
           'transpose',
           'reshape_from_spec',
           'build_run_spec']

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
        :flat: an MLDataset
        :raise_err: raise or not
    Returns:
        :bool: ``True`` if flat ``False`` or ``ValueError`` if not flat (raise_err=True)
    '''
    arr = getattr(dset, features_layer, None)
    if arr is None or not hasattr(arr, 'dims') or tuple(arr.dims) !=  FEATURES_LAYER_DIMS:
        msg = 'Expected an MLDataset/Dataset with DataArray "{}" and dims {}'
        if raise_err:
            raise ValueError(msg.format(features_layer, FEATURES_LAYER_DIMS))
        else:
            return False
    return True


def _on_each_layer_arr_step(namespace, layer, arr, step_label, parts):
    coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
    attrs = copy.deepcopy(arr.attrs)
    if 'agg' == step_label:
        func, args, kw = parts
        if not callable(func):
            func = getattr(arr, func)
        arr = func(*args, **kw)
    elif 'flatten' == step_label:
        new_dim, trans_dims = parts
        if tuple(trans_dims) != tuple(arr.dims):
            arr = transpose(arr, trans_dims)
            coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
            attrs = copy.deepcopy(arr.attrs)
        index = create_multi_index(arr)
        val = val.ravel()[:, np.newaxis]
        coords = OrderedDict([(new_dim, index), ('layer', [layer])])
        new_dims_2 = (new_dim, 'layer')
        arr = xr.DataArray(val, coords=coords, dims=new_dims_2, attrs=attrs)
    namespace[layer] = arr
    return arr


def _concat_arrs(namespace, new_layer, step_label, parts, using_layers, concat_dim='layer'):
    arrs = []
    for layer in using_layers:
        arr = namespace[layer]
        args = namespace, layer, arr, step_label, parts
        arr = _on_each_layer_arr_step(*args)
        arrs.append(arr)
    if 'flatten' == step_label:
        if not isinstance(new_layer, tuple):
            if all(concat_dim in arr.dims for arr in arrs):
                new_arr = xr.concat(arrs, dim=concat_dim)
                return new_arr
    return namespace


def _calc_new_layer_one_step(namespace, step, new_layer, using_layers):
    from xarray_filters.ml_features import MLDataset
    step_label, parts = step[0], list(step[1:])
    new_dset = namespace
    if step_label in namespace:
        namespace[step_label] = namespace[step_label].transpose(*parts[1])
    elif 'layers' == step_label:
        arrs = (namespace[layer] for layer in using_layers)
        new_dset = MLDataset(OrderedDict(zip(using_layers, arrs)))
    elif 'transform' == step_label:
        func, args, kw = parts
        kw.update(namespace)
        new_dset = func(*args, **kw)
    else:
        args = namespace, new_layer, step_label, parts, using_layers
        return _concat_arrs(*args)
    return new_dset


def _calc_new_layer(spc, dset, new_layer, namespace, verbose=True):
    using_layers = None
    new_dset = None
    for idx, step in enumerate(spc):
        if 'layers' == step[0]:
            using_layers = step[1]
        elif hasattr(dset, 'data_vars') and using_layers is None:
            using_layers = tuple(dset.data_vars)
        args = namespace, step, new_layer, using_layers
        new_dset = _calc_new_layer_one_step(*args)
        new_dset = new_dset
        if isinstance(new_dset, xr.DataArray):
            namespace[new_layer] = new_dset
        elif hasattr(new_dset, 'data_vars'):
            namespace.update(new_dset.copy(deep=True).data_vars)
        elif hasattr(new_dset, 'items'):
            namespace.update(new_dset)
    return namespace


@dask.delayed
def reshape_from_spec(spec, dset,
                      keep_existing_arrs=True,
                      verbose=True, copy=True,
                      return_dict=False):
    from xarray_filters.ml_features import MLDataset
    print('Spec', spec) # TODO remove
    if not hasattr(spec, 'items'):
        spec = OrderedDict(spec)
    new_dset = OrderedDict()
    namespace = OrderedDict(dset.data_vars)
    for new_layer, spc in spec.items():
        namespace = _calc_new_layer(spc, dset, new_layer, namespace, verbose=True)
        if not isinstance(new_layer, tuple):
            new_layers = (new_layer,)
        else:
            new_layers = new_layer
        for layer in new_layers:
            new_dset[layer] = namespace[layer]
    dv = OrderedDict()
    if keep_existing_arrs:
        if copy:
            dset = dset.copy(deep=True)
        dv.update(dset.data_vars)
    if return_dict:
        return new_dset
    return MLDataset(new_dset)


def _format_aggs_transforms(agg_or_trans, action='agg'):
    if agg_or_trans is None:
        new_agg_or_trans = []
    if callable(agg_or_trans):
        new_agg_or_trans = [[action, agg_or_trans, [], {}]]
    elif isinstance(agg_or_trans, (tuple, list)):
        agg_or_trans = list(agg_or_trans)
        if not isinstance(agg_or_trans[0], (tuple, list)):
            agg_or_trans = [agg_or_trans]
        new_agg_or_trans = []
        for agg in agg_or_trans:
            if callable(agg):
                new_agg_or_trans.append([action, agg, [], {}])
            elif isinstance(agg, (list, tuple)):
                agg = list(agg)
                if len(agg) == 4:
                    continue
                else:
                    args = [_ for _ in agg if isinstance(_, (tuple, list))]
                    kw = [_ for _ in agg if isinstance(_, dict)]
                    func = [_ for _ in agg if isinstance(_, str) or callable(_)]
                    if not func:
                        raise ValueError('Expected a string DataArray method name or a callable in {}'.format(agg))
                    func = func[0]
                    if args:
                        args = args[0]
                    if kw:
                        kw = kw[0]
                    else:
                        kw = dict()
                    agg = [action, func, args, kw]
            new_agg_or_trans.append(agg)
    return new_agg_or_trans


def build_run_spec(dset, name=None, layers=None,
                   aggs=None, flatten=False,
                   copy=True, transforms=None,
                   return_dict=False,
                   keep_existing_arrs=True):
    '''Check if an MLDataset has a DataArray called "features"
    with dimensions (space, layer)
    Parameters:
        :dset: an MLDataset or xarray.Dataset instance
        :name: name of a new layer to be added to dset
        :layers: names of layers (DataArrays) needed for aggregations,
                 transforms, or flatten operations that make a new
                 layer.  If layers is None, then all DataArrays are
                 used.
        :aggs: Controls aggregation (reduction) operations to reduce
               dimensionality of each layer in layers
        :flatten: False
        :copy: True
        :transforms:None
        :return_dict:False
        :keep_existing_arrs:True
        :raise_err: raise or not
    Returns:
        :bool: ``True`` if flat ``False`` or ``ValueError`` if not flat (raise_err=True)
    '''
    transforms = _format_aggs_transforms(transforms, action='transform')
    aggs = _format_aggs_transforms(aggs)
    layers = [['layers', layers or list(dset.data_vars)]]
    name = name or tuple(layers[0][1])
    if flatten is None:
        flatten = [['flatten', 'space', ['y', 'x']]]
    elif flatten is False:
        flatten = []
    else:
        flatten = list(flatten)
        if 'flatten' not in flatten:
            flatten = [['flatten'] + flatten]
        else:
            flatten = [flatten]
    spec = [(name, layers + transforms + aggs + flatten)]
    return reshape_from_spec(spec, dset,
                             return_dict=return_dict,
                             keep_existing_arrs=keep_existing_arrs,
                             copy=copy)


def concat_ml_features(*dsets,
                       features_layer=FEATURES_LAYER,
                       concat_dim=None,
                       keep_attrs=False): # TODO True or False (convention?)
    from xarray_filters.ml_features import MLDataset
    if keep_attrs:
        attrs = dset.attrs.copy()
    else:
        attrs = {}
    concat_dim = concat_dim or FEATURES_LAYER_DIMS[1]
    data_arrs = []
    for dset in dsets:
        if not isinstance(dset, MLDataset):
            raise ValueError('TODO -error message here')
        data_arr = dset.data_vars.get(features_layer, None)
        if data_arr is None:
            raise ValueError('TODO -error message here')
        data_arrs.append(data_arr)
    data_arr = xr.concat(data_arrs, dim=concat_dim)
    return MLDataset(OrderedDict([(features_layer, data_arr)]), attrs=attrs)

