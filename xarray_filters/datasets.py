"""
This module contains a routines to convert the output of

    sklearn.datasets.make_blobs
    sklearn.datasets.make_regression
    sklearn.datasets.make_classification

to the following containers

    xarray.Dataset
    numpy.array
    pandas.DataFrame

The sklearn functions all output a tuple (X, y) where

    X is a n_samples by n_features numpy.array of features
    y is a n_samples numpy.array of labels

The reasons to wrap the sklearn routines are:

- We want new types easily (xarray.Dataset, etc.)
- We want multidimensional features.
- We want to generate sample weights together with synthetic data.

Motivating usecases:

- Generate data to test models that predict temperature or humidity at
  3d-spatial coordinates and time based on earlier measurements at nearby/same
  coordinates.

Implementation Notes
--------------------

The final routines should be drop-in replacements for the sklearn functions,
but with extra keyword arguments to address the needs outlined above. The
public interface of this module should be only the `make_*` routines; all other
names should have a leading underscore.

All exposed functions should have useful help strings, and an informative
signature (no *args, **kwargs).


"""



from collections import Sequence, OrderedDict
from functools import partial, wraps

import numpy as np
import sklearn.datasets as skdatasets
import xarray as xr
import inspect


NON_SKLEARN_KWARGS = ('x_output_type',
                      'y_output_type',
                      'shape',
                      'dims',
                      'as_flat',
                      'reshaper',
                      'make_sample_weight',)

DEFAULT_SHAPE = (50, 50)


def flexible_out(skl_make_data_func):
    """Add keyword arguments to skl_make_data_func.

    Note: a decorator doesn't solve this. It could add the functionality, but
    it would not change the docstring or the signature of the function, as it
    would keep the original one from the wrapped function from sklearn. So we
    have to do the process manually.
    """
    def wrapper(*args, astype=None, **kwargs):
        X, y = skl_make_data_func(*args, **kwargs)
        if astype:
            Xcast = astype(X)
            ycast = astype(y) 
            return Xcast, ycast
        else:
            return X, y

    # We now have two tasks: (1) fix the docstring and (2) fix the signature of `wrapper`.
    # Task 1 of 2: fixing the docstring of `wrapper`
    preamble_doc = """Like {funcname}, but with extra keyword parameters.

    Extra Parameters
    ----------------
    astype: type
        one of pandas.DataFrame, xarray.Dataset, numpy.array. Has to be passed
        by keyword.

    Returns
    -------
    X: feature matrix
        Like {funcname}, but with type given by astype.
    y: labels vector
        Like {funcname}, but with type given by astype.

    See the docs of the sklearn function below
    ---

    """.format(funcname=('sklearn.datasets.' + skl_make_data_func.__code__.co_name))
    wrapper.__doc__ = preamble_doc + skl_make_data_func.__doc__
    # Task 2 of 2: fixing the signature of `wrapper`
    argspec = inspect.getfullargspec(skl_make_data_func)
    params = [inspect.Parameter(name=pname, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, default=pdefault)
              for (pname, pdefault) in zip(argspec.args, argspec.defaults)]
    params += [inspect.Parameter(name='astype',
        kind=inspect.Parameter.KEYWORD_ONLY, default=None)]
    wrapper.__signature__ = inspect.Signature(params)
    wrapper.__name__ = skl_make_data_func.__name__
    return wrapper

make_blobs = flexible_out(skldatasets.make_blobs)

# [WIP] Try this: 
#
# [WIP]     import pandas as pd
# [WIP]     help(make_blobs)   # note argument list and docstring
# [WIP]     make_blobs(astype=pd.DataFrame)
#
# [WIP] This shows where we are going. The idea is now to extend that to extra
# [WIP] keyword arguments. The challenge is to keep the code readable too. Right now,
# [WIP] the resulting module that is imported is more understandable than the code
# [WIP] itself.
