from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import xarray as xr

from xarray_filters import *
from xarray_filters.tests.test_data import *

def test_once():
    X = new_test_dataset(TEST_LAYERS)
    new_X = X.new_layer(name='tp1',
                        layers=['temperature', 'pressure'],
                        aggs=[['mean', dict(dim='z')],
                              ['std', dict(dim='t')]],
                        flatten=None,
                        return_dict=False).compute()

    new_Xb = X.new_layer(name=None,
                         layers=['temperature', 'pressure'],
                         aggs=['mean', dict(dim='z')],
                         flatten=False
                         ).compute().new_layer(
                           name='tp2',
                           layers=['temperature', 'pressure'],
                           aggs=['std', dict(dim='t')],
                           flatten=['space', ['y', 'x']]
                        ).compute()
    assert np.all(new_Xb.tp2 == new_X.tp1)

def test_transform_no_name():
    '''with no "name" keyword - all layers are returned'''
    X = new_test_dataset(TEST_LAYERS)
    example = X.new_layer(
        name=None,
        layers=None,
        transforms=iqr_standard,
        flatten=False,
    ).compute()
    assert isinstance(example, MLDataset)
    assert list(example.data_vars) == TEST_LAYERS
    for key, data_arr in example.data_vars.items():
        assert key in X.data_vars
        assert X[key].dims == data_arr.dims


def test_aggs_no_name():
    '''with no "name" keyword, all layers are passed through
    but aggs changes the dimensionality'''
    X = new_test_dataset(TEST_LAYERS)
    example = X.new_layer(
        name=None,
        layers=None,
        aggs=['quantile', (0.5,), dict(dim=('t', 'z'))],
        transforms=iqr_standard,
        flatten=False,
    ).compute()
    assert isinstance(example, MLDataset)
    assert list(example.data_vars) == TEST_LAYERS
    for key, data_arr in example.data_vars.items():
        assert key in X.data_vars
        assert data_arr.dims == ('x', 'y')

def test_named_aggregation_to_features():
    X = new_test_dataset(TEST_LAYERS)
    name = 'new_data_array'
    example = X.new_layer(
        name=name,
        layers=None,
        aggs=['quantile', (0.5,), dict(dim=('t', 'z'))],
        transforms=iqr_standard,
        flatten=None,
    ).compute()
    assert isinstance(example, MLDataset)
    assert name not in X.data_vars
    assert tuple(example.data_vars) == (name,)
    assert example[name].dims == ('space', 'layer')
    assert list(example[name].layer.values) == TEST_LAYERS
    assert len(tuple(example[name].space.values[0])) == 2 # (x, y)

def test_to_and_from_feature_matrix():
    X = new_test_dataset(TEST_LAYERS).mean(dim=('z', 't'))
    X2 = X.to_ml_features().compute()
    assert (FEATURES_LAYER,) == tuple(X2.data_vars)
    X3 = X2.from_ml_features()
    for layer, data_arr in X3.data_vars.items():
        assert layer in X.data_vars
        assert np.all(X[layer].values == X3[layer].values)


