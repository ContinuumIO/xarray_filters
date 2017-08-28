'''
-----------------------
``earthio.reshape``
~~~~~~~~~~~~~~~~~~~~~~~
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple, OrderedDict
from copy import deepcopy
import os
import warnings

import attr
import dask
import numpy as np
import pandas as pd
import scipy.interpolate as spi
import xarray as xr

from xarray_filters.constants import FEATURES_LAYER_DIMS, FEATURES_LAYER
from xarray_filters.multi_index import create_multi_index
from xarray_filters.pipe_utils import for_each_array, call_custom_func

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
        :dset: an MLDataset
        :raise_err: raise or not
    Returns:
        :bool: ``True``, ``False`` or raises ``ValueError``
               if not flat (raise_err=True)
    '''
    arr = getattr(dset, features_layer, None)
    if arr is None or not hasattr(arr, 'dims') or tuple(arr.dims) !=  FEATURES_LAYER_DIMS:
        msg = 'Expected an MLDataset/Dataset with DataArray "{}" and dims {}'
        if raise_err:
            raise ValueError(msg.format(features_layer, FEATURES_LAYER_DIMS))
        else:
            return False
    return True


def _on_each_layer_arr_step(namespace, layer, arr, step_label, parts, features_layer_dims=None):

    if features_layer_dims is None:
        col_dim = FEATURES_LAYER_DIMS[1]
    else:
        col_dim = features_layer_dims[1]
    coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
    attrs = deepcopy(arr.attrs)
    if 'flatten' == step_label:
        new_dim, trans_dims = parts
        if trans_dims is not None and tuple(trans_dims) != tuple(arr.dims):
            arr = transpose(arr, trans_dims)
            coords, dims, val, attrs = arr.coords, arr.dims, arr.values, arr.attrs
            attrs = deepcopy(arr.attrs)
        if len(dims) == 1:
            new_dim = dims[0]
            index = getattr(arr, new_dim)
        else:
            index = create_multi_index(arr)
        val = val.ravel()[:, np.newaxis]
        coords = OrderedDict([(new_dim, index), (col_dim, [layer])])
        new_dims_2 = (new_dim, col_dim)
        arr = xr.DataArray(val, coords=coords, dims=new_dims_2, attrs=attrs)
    namespace[layer] = arr
    return arr


def _same_size_dims_arrs(*arrs, raise_err=True):
    '''Check if all DataArrays in arrs have same size and same dims

    Parameters:
        :raise_err: If True, raise ValueError if dims/sizes differ
                    else return True/False'''
    siz = None
    dims = None
    for arr in arrs:
        if siz is not None and siz != arr.size:
            if raise_err:
                raise ValueError('Expected arrays of same size but found {} and {}'.format(siz, arr.size))
            return False
        if dims is not None and tuple(arr.dims) != dims:
            if raise_err:
                raise ValueError('Expected arrays of same dims but found {} and {}'.format(dims, arr.dims))
            return False
    return dims, siz


def _concat_arrs(namespace, new_layers, step_label, parts, using_layers,
                 concat_dim=None):
    from xarray_filters.ml_features import MLDataset
    arrs = []
    for layer in using_layers:
        if layer in namespace:
            arr = namespace[layer]
            args = namespace, layer, arr, step_label, parts
            arr = _on_each_layer_arr_step(*args)
            arrs.append(arr)
    if 'transform' == step_label:
        func, args, kwargs = parts
        #print('transform', func, args, kwargs)
        if callable(func):
            kw = OrderedDict()
            kw.update(kwargs)
            kw.update(namespace)
            return call_custom_func(func, *args, layers_list=namespace, **kw)
        else:
            new_dset = OrderedDict()
            for layer, arr in namespace.items():
                if isinstance(arr, (xr.DataArray, xr.Dataset, MLDataset)):
                    func_handle = getattr(arr, func)
                    namespace[layer] = call_custom_func(func_handle, *args, layers_list=namespace, **kwargs)
    elif 'flatten' == step_label:
        new_layer = tuple(set(new_layers) - set(namespace))
        if new_layer and len(new_layer) == 1:
            dims, siz = _same_size_dims_arrs(*arrs)
            concat_dim = concat_dim or FEATURES_LAYER_DIMS[1]
            if all(concat_dim in arr.dims for arr in arrs):
                new_arr = xr.concat(arrs, dim=concat_dim)
                return new_arr
        else:
            raise ValueError('TODO - document how one ends up here :) {}'.format((new_layers, namespace.keys(), using_layers)))
    return namespace


def _calc_new_layer_one_step(namespace, step, new_layers, using_layers):
    from xarray_filters.ml_features import MLDataset
    step_label, parts = step[0], list(step[1:])
    new_dset = None
    if step_label in namespace:
        namespace[step_label] = namespace[step_label].transpose(*parts[1])
    elif 'layers' == step_label:
        arrs = (namespace[layer] for layer in using_layers
                if layer in namespace)
        new_dset = MLDataset(OrderedDict(zip(using_layers, arrs)))
    else:
        args = namespace, new_layers, step_label, parts, using_layers
        return _concat_arrs(*args)
    return new_dset


def _update_namespace(new_layer, new_dset, namespace):
    #print('un', new_layer, type(new_dset), namespace.keys())
    if isinstance(new_dset, xr.DataArray):
        if new_layer is not None:
            namespace[new_layer] = new_dset
        return namespace
    if hasattr(new_dset, 'data_vars'):
        as_dict = new_dset.copy(deep=True).data_vars
    elif hasattr(new_dset, 'items'):
        as_dict = new_dset
    else:
        raise ValueError(repr((new_layer, new_dset, namespace.keys())))
    if new_layer in as_dict:
        _update_namespace(new_layer, as_dict[new_layer], namespace)
    else:
        for k, arr in as_dict.items():
            _update_namespace(k, arr, namespace)
    return namespace


def _calc_new_layer(spc, dset, new_layers, namespace, verbose=True):
    using_layers = None
    new_dset = None
    for idx, step in enumerate(spc):
        if 'layers' == step[0]:
            using_layers = tuple(step[1])
        elif hasattr(dset, 'data_vars') and using_layers is None:
            using_layers = tuple(dset.data_vars)
        new = tuple(layer for layer in new_layers if layer not in using_layers)
        using_layers = using_layers + new
        args = namespace, step, new_layers, using_layers
        new_dset = _calc_new_layer_one_step(*args)
        for new_layer in new_layers:
            _update_namespace(new_layer, new_dset, namespace)
    return namespace


@dask.delayed
def reshape_from_spec(spec, dset,
                      keep_existing_layers=True,
                      verbose=True, copy=True,
                      return_dict=False):
    from xarray_filters.ml_features import MLDataset
    print('Spec passed to reshape_from_spec', spec) # TODO - remove this later, but for now it
                        # shows how the spec looks for given args
    if not hasattr(spec, 'items'):
        spec = OrderedDict(spec)
    new_dset = OrderedDict()
    namespace = OrderedDict(dset.data_vars)
    original_layers = tuple(namespace)
    for new_layer, spc in spec.items():
        if not isinstance(new_layer, (tuple, list,)):
            new_layers = (new_layer,)
        else:
            new_layers = new_layer
        namespace = _calc_new_layer(spc, dset, new_layers,
                                    namespace, verbose=True)
    for layer, arr in namespace.items():
        if layer not in original_layers or keep_existing_layers:
            #print(layer, 'is', type(arr))
            if isinstance(arr, xr.DataArray):
                new_dset[layer] = namespace[layer]
    if return_dict:
        return new_dset
    return MLDataset(new_dset)


def _format_transforms(trans, action='transform'):
    '''TODO - Document the list of list structures
    that can be passed transforms.  See
    comments in tests/test_reshape.py for now

    Parameters:
        :trans: "transforms" arguments to build_run_spec
        :action:       "transform"

    Returns:
        list of lists structure for reshape_from_spec
    '''
    if trans is None:
        return []
    if callable(trans):
        new_trans = [[action, trans, [], {}]]
    elif isinstance(trans, (tuple, list)):
        trans = list(trans)
        new_trans = []
        for tran in trans:
            if callable(tran):
                new_trans.append([action, tran, [], {}])
            elif isinstance(tran, (list, tuple)):
                tran = list(tran)
                args = [_ for _ in tran if isinstance(_, (tuple, list))]
                kw = [_ for _ in tran if isinstance(_, dict)]
                func = [_ for _ in tran if isinstance(_, str) or callable(_)]
                if not func:
                    raise ValueError('Expected a string DataArray method name or a callable in {}'.format(tran))
                func = func[0]
                if args:
                    args = args[0]
                if kw:
                    kw = kw[0]
                else:
                    kw = dict()
                tran = [action, func, args, kw]
                new_trans.append(tran)
    return new_trans


def build_run_spec(dset, name=None, layers=None, flatten=False,
                   copy=True, transforms=None,
                   return_dict=False,
                   keep_existing_layers=True,
                   compute=None):
    '''Check if an MLDataset has a DataArray called "features"
    with dimensions (space, layer)
    Parameters:
        :dset: an MLDataset or xarray.Dataset instance
        :name: name of a new layer to be added to dset
        :layers: names of layers (DataArrays) needed for aggregations,
                 transforms, or flatten operations that make a new
                 layer.  If layers is None, then all DataArrays are
                 used.
        :flatten: False
        :copy: True
        :transforms: None
        :return_dict: False
        :keep_existing_layers: True
        :compute: If False/None, do not call .compute
                  If True call .compute
                  If a dict, then pass kwargs, e.g. dset.compute(**compute)
                  TODO - copy this help to all funcs
                  using compute=None as keyword
    Returns:
        :bool: ``True`` if flat ``False`` or ``ValueError`` if not flat (raise_err=True)
    '''
    transforms = _format_transforms(transforms)
    if layers is None:
        layers = dset.data_vars
    layers = tuple(layers)

    if flatten is True:
        # Typically "space" for the new row dimension
        # None means no transpose before .ravel
        flatten = [['flatten', FEATURES_LAYER_DIMS[0], None]]
    elif flatten is False:
        flatten = []
    else:
        flatten = list(flatten)
        if 'flatten' not in flatten:
            flatten = [['flatten'] + flatten]
        else:
            flatten = [flatten]
    if flatten and not name:
        name = (FEATURES_LAYER,)
    if flatten:
        keep_existing_layers = False
    layers_step = [['layers', layers]]
    spec = [(name, layers_step + transforms + flatten)]
    dset = reshape_from_spec(spec, dset,
                             return_dict=return_dict,
                             keep_existing_layers=keep_existing_layers,
                             copy=copy)
    if compute:
        if not isinstance(compute, dict):
            compute = dict()
        return dset.compute(**compute)
    return dset


def concat_ml_features(*dsets,
                       features_layer=FEATURES_LAYER,
                       concat_dim=None,
                       keep_attrs=False):

    '''Concatenate MLDataset / Dataset (dsets) along concat_dim
    (by default the column dimension, typically called "layer")

    Parameters:
        :dsets: Any number of MLDataset / Dataset objects that are
                2D
        :features_layer: Typically "layer", the column dimension
        :concat_dim: If None, the column dimension is guessed
        :keep_attrs: If True, keep the attrs of the first dset in *dsets
    TODO - Gui: This could use the astype logic discussed elsewhere?


    '''

    # TODO True or False (convention?)
    from xarray_filters.ml_features import MLDataset
    if not dsets:
        raise ValueError('No MLDataset / Dataset arguments passed.  Expected >= 1')
    if keep_attrs:
        attrs = dsets[0].attrs.copy()
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

