from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from xarray_filters.datasets import *
from xarray_filters.mldataset import MLDataset


def test_make_blobs():
    blobs1 = make_blobs()
    assert isinstance(blobs1, MLDataset)  # check default type
    shape2 = (20, 4)
    blobs2 = make_blobs(n_samples=100, shape=shape2)
    for k in blobs2.keys():
        assert blobs2[k].shape == shape2

