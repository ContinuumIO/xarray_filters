from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
from itertools import product

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_filters import *
from xarray_filters.tests.test_data import *

def test_aggregations_can_chain():
    X = new_test_dataset(TEST_LAYERS)
    # The following should be the same: new_X and new_Xb
    args_kwargs = [[MLDataset.mean, dict(dim='z')], # From 4D to 3D
                   [MLDataset.std, dict(dim='t')]] # From 3D to 2D
    layers = ['temperature', 'pressure']
    new_X = X.chain(args_kwargs, layers=layers
                   ).to_features(features_layer='temp_pres')
    new_Xb = X.chain(args_kwargs[:1],
                     layers=layers
                     ).chain(args_kwargs[1:],
                             layers=layers
                             ).to_features(
                             features_layer='temp_pres')
    assert new_Xb.temp_pres.ndim == 2
    assert new_X.temp_pres.ndim == 2
    assert np.all(new_Xb.temp_pres == new_X.temp_pres)


def test_transform_no_name():
    '''with no "name" keyword - all layers are returned'''
    X = new_test_dataset(TEST_LAYERS)
    example = X.chain(iqr_standard)
    assert isinstance(example, MLDataset)
    assert list(example.data_vars) == TEST_LAYERS
    for key, data_arr in example.data_vars.items():
        assert key in X.data_vars
        assert X[key].dims == data_arr.dims


def test_transforms_no_name():
    '''with no "name" keyword, all layers are passed through
    but transforms changes the dimensionality'''
    X = new_test_dataset(TEST_LAYERS)
    step_1 = [MLDataset.quantile, 0.5, dict(dim=('t', 'z'))]
    example = X.chain([step_1, iqr_standard])
    assert isinstance(example, MLDataset)
    assert list(example.data_vars) == TEST_LAYERS
    for key, data_arr in example.data_vars.items():
        assert key in X.data_vars
        assert data_arr.dims == ('x', 'y')


def test_named_aggregation_to_features():
    X = new_test_dataset(TEST_LAYERS)
    name = 'new_data_array'
    step_1 = iqr_standard
    step_2 = [MLDataset.quantile, 0.5, dict(dim=('t', 'z'))]
    example = X.chain((step_1, step_2)).to_features(features_layer=name)
    assert isinstance(example, MLDataset)
    assert name not in X.data_vars
    assert tuple(example.data_vars) == (name,)
    assert example[name].dims == ('space', 'layer')
    assert list(example[name].layer.values) == TEST_LAYERS
    assert len(tuple(example[name].space.values[0])) == 2 # (x, y)


def test_to_and_from_feature_matrix():
    X = new_test_dataset(TEST_LAYERS).mean(dim=('z', 't'))
    X2 = X.to_features()
    assert (FEATURES_LAYER,) == tuple(X2.data_vars)
    arr = X2[FEATURES_LAYER]
    assert arr.shape[1] == len(X.data_vars)
    assert arr.shape[0] == X.temperature.size
    X3 = X2.from_features()
    for layer, data_arr in X3.data_vars.items():
        assert layer in X.data_vars
        assert np.allclose(X[layer].values, X3[layer].values)


def test_data_vars_keywords_varkw():
    X = new_test_dataset(TEST_LAYERS)
    name = 'magnitude'
    @data_vars_func
    def example(**kw):
        for layer, arr in kw.items():
            if isinstance(arr, xr.DataArray):
                assert layer in X.data_vars
                assert isinstance(X[layer], xr.DataArray)
        mag = (kw['wind_x'] ** 2 + kw['wind_y'] ** 2) ** 0.5
        return MLDataset(OrderedDict([(name, mag)]))
    X2 = X.chain([example])
    assert isinstance(X2, MLDataset)
    assert 'magnitude' in X2.data_vars


@pytest.mark.parametrize('layers', (None, ('wind_x', 'wind_y')))
def test_data_vars_keywords_positional(layers):
    X = new_test_dataset(TEST_LAYERS)
    @data_vars_func
    def example(wind_x, wind_y, temperature, pressure, **kw):
        degree = kw.get('degree')
        assert degree is not None
        mag = (wind_x ** degree + wind_y ** degree) ** (1 / degree)
        layers_with_mag = tuple(TEST_LAYERS) + ('magnitude',)
        arrs = (wind_x, wind_y, temperature, pressure, mag)
        return OrderedDict(zip(layers_with_mag, arrs))

    @for_each_array
    def coef_of_variation(arr, dim=None):
        return arr.std(dim=dim) / arr.mean(dim=dim)
    X2 = X.chain([(example, dict(degree=2)),
                  (coef_of_variation,
                   dict(dim=('z', 't')))])
    assert isinstance(X2, MLDataset)
    assert 'magnitude' in X2.data_vars
    if layers is not None:
        assert all(layer in X2.data_vars for layer in TEST_LAYERS)
    else:
        assert len(X2.data_vars)

Xs = [
    MLDataset({'pressure': xr.DataArray(np.random.uniform(0, 1, (2,3)),
                                        coords={'x': np.arange(2),
                                                'y': np.arange(3)},
                                        dims=['x', 'y']),
               'temperature': xr.DataArray(np.random.uniform(0, 1, (2,3)),
                                           coords={'x': np.arange(2),
                                                   'y': np.arange(3)},
                                           dims=['x', 'y'])}),
    MLDataset({'pressure': xr.DataArray(np.random.uniform(0, 1, 6),
                                        coords={'x': np.arange(6)},
                                        dims=['x']),
               'temperature': xr.DataArray(np.random.uniform(0, 1, 6),
                                           coords={'x': np.arange(6)},
                                           dims=['x'])}),
]
@pytest.mark.parametrize('X', Xs)
def test_from_features_dropped_rows(X):
    features = X.to_features()
    data1 = features.from_features()

    # Assert that we get the original Dataset back after X.to_features().from_features()
    assert np.array_equal(data1.coords.to_index().values, X.coords.to_index().values)
    assert np.allclose(data1.to_xy_arrays()[0], X.to_xy_arrays()[0])

    # Drop some rows
    features['features'].values[:2, :] = np.nan
    zerod_vals_copy = features['features'].values[:] # Copy NaN positions for testing later on
    features = features.dropna(features['features'].dims[0])

    # Convert back to original dataset, padding NaN values into the proper locations if necessary
    data2 = features.from_features()

    # Assert that the coords are correct, and NaNs are in the right places
    if np.nan in data2.to_xy_arrays()[0]:
        assert np.array_equal(data2.coords.to_index().values, data1.coords.to_index().values)
        assert np.allclose(data2.to_xy_arrays()[0], zerod_vals_copy, equal_nan=True)

dsets = [xr.Dataset(x) for x in Xs]
np_arrs = [x.to_features().features.values for x in Xs]
dfs = [pd.DataFrame(x.to_features().features.values) for x in Xs]
choices = {
    'mldataset': Xs,
   # 'dataset': dsets,
   # 'numpy': np_arrs,
   # 'dataframe': dfs,
}

y_types = ('numpy', )# 'mldataset', dataset', 'numpy', 'none',)
@pytest.mark.parametrize('X_key, y_type, idx', product(choices, y_types, range(len(Xs))))
def test_to_xy_arrays(X_key, y_type, idx):
    X = choices[X_key][idx]
    shp = to_features(X).features.shape
    rows = shp[0]
    y = np.arange(rows)[:, np.newaxis]
    if y_type in ('mldataset', 'dataset'):
        y = to_features(X).features.values.sum(axis=1)[:, np.newaxis]

        coords = {'space': np.arange(y.shape[0]), 'layer': ['y']}
        arr = xr.DataArray(y,
                           coords=coords,
                           dims=('space', 'layer'))
        y = MLDataset(OrderedDict([('y', arr)]))
        if y_type == 'dataset':
            y = xr.Dataset(y)
    elif y_type == 'none':
        y = None
    if X is not None and hasattr(X, 'to_xy_arrays'):
        X2, y2 = X.to_xy_arrays(y)
        X3, y3 = to_xy_arrays(X, y)
        assert (X2 is None and X3 is None) or (np.allclose(X2, X3))
        assert (y2 is None and y3 is None) or (np.allclose(y2, y3))
    else:
        X2, y2 = to_xy_arrays(X, y)
    if X is not None:
        assert isinstance(X2, np.ndarray)
    else:
        assert X2 is None
    if y is None:
        assert y2 is None
    else:
        assert isinstance(y2, np.ndarray)
