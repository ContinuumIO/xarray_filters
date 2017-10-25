from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from xarray_filters.datasets import *
from xarray_filters.mldataset import MLDataset


def test_make_blobs():
    # Test default type
    blobs1 = make_blobs(n_samples = 12, n_features=4) # TODO: add kwarg: chunks=2
    assert isinstance(blobs1, MLDataset)
    # Test shape precedence over n_samples
    params2 = dict(shape=(20, 4), n_samples=100) # TODO: add kwarg: chunks=4
    blobs2 = make_blobs(**params2)
    assert np.prod(params2['shape']) != params2['n_samples']
    for k in blobs2.keys():
        assert blobs2[k].shape == params2['shape']
    # Test if dimensions, layers and labels (yname) can have custom names.
    params3 = dict(dims=list('xyzt'), shape=(3,3,3,3), layers=['temp',
        'pressure'], yname='target') # TODO: add kwarg: chunks=3
    blobs3 = make_blobs(**params3)
    assert set(blobs3.keys()) == set(params3['layers']).union({params3['yname']})
    dimshape3 = dict(zip(params3['dims'], params3['shape']))
    for k in blobs3.dims:
        assert blobs3.dims[k] == dimshape3[k]
