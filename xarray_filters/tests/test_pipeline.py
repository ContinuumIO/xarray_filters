from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from xarray_filters.pipeline import Pipeline, Step, WriteNetCDF

@pytest.fixture
def ds_under_test():
    from xarray_filters.tests.test_data import new_test_dataset
    ds = new_test_dataset(('wind', 'pressure', 'temperature'))
    return ds

def test_set_params_fit_xform(ds_under_test):
    def step_1(dset, **kw):
        return kw['a'] * dset.mean(dim=('x', 'y')) ** kw['b']

    def step_2(dset, **kw):
        return kw['a'] + dset * kw['b']

    class Generic(Step):
        a = 0
        b = 1
        func = None
        def transform(self, dset):
            params = self.get_params()
            dset = self.func(dset=dset, **params)
            return dset

    steps = (('s1', Generic(func=step_1)),
             ('s2', Generic(func=step_2)),
             ('s3', WriteNetCDF(fname='two_step_pipeline_out.nc')))
    pipe = Pipeline(steps=steps)
    ((_, pipe_step1), (_, pipe_step2), (_, pipe_step3)) = steps

    assert pipe_step1.a == 0 and pipe_step1.b == 1 and pipe_step1.func is step_1
    assert pipe_step2.a == 0 and pipe_step2.b == 1 and pipe_step2.func is step_2
    assert pipe_step3.fname == 'two_step_pipeline_out.nc'

    pipe.set_params(s1__a=2,
                    s1__b=3,
                    s2__a=0,
                    s2__b=0,
                    s3__fname='file_with_zeros.nc')

    assert pipe_step1.a == 2 and pipe_step1.b == 3 and pipe_step1.func is step_1
    assert pipe_step2.a == 0 and pipe_step2.b == 0 and pipe_step2.func is step_2
    assert pipe_step3.fname == 'file_with_zeros.nc'

    pipe.transform(ds_under_test)

def test_estimator_cloning(ds_under_test):
    from sklearn.base import clone

    class Generic(Step):
        a = 10
        b = 12
        func = None
        lst = []
        def transform(self, dset):
            params = self.get_params()
            dset = self.func(dset=dset, **params)
            return dset

    def step_1(dset, **kw):
        return kw['a'] * dset.mean(dim=('x', 'y')) ** kw['b']

    g_estimator = Generic(func=step_1, lst=[[1], 2, 3])
    g_estimator_clone = clone(g_estimator)

    assert g_estimator.a == g_estimator_clone.a
    assert g_estimator.b == g_estimator_clone.b
    assert g_estimator.func == g_estimator_clone.func
