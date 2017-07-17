from collections import Sequence, OrderedDict
from functools import partial

import numpy as np
import sklearn.datasets as skdatasets
import xarray as xr


NON_SKLEARN_KWARGS = ('x_output_type',
                      'y_output_type',
                      'shape',
                      'dims',
                      'as_flat',
                      'reshaper',
                      'make_sample_weight',)

DEFAULT_SHAPE = (50, 50)

ds = dir(skdatasets)

_datasets = {f: getattr(skdatasets, f)
             for f in ds if f[0] != '_' and f != 'samples_generator'}
_make_funcs = {k: v for k, v in _datasets.items() if k.startswith('make_')}
_load_funcs = {k: v for k, v in _datasets.items() if k.startswith('load_')}
_fetch_funcs = {k: v for k, v in _datasets.items() if k.startswith('fetch_')}


OUTPUT_TYPES = {'numpy:ndarray': ['numpy', 'ndarray',
                                  'numpy:ndarray', np.ndarray],
                'xarray:Dataset': ['Dataset', 'xarray:Dataset', xr.Dataset]}

def _find_type(output_type):
    for k, v in OUTPUT_TYPES.items():
        if output_type in v:
            return k

def _split_sklearn_out(sklearn_out, make_sample_weight=None, **kwargs):
    if isinstance(sklearn_out, tuple) and len(sklearn_out) == 2:
        X, y = sklearn_out
    else:
        X = sklearn_out
        y = None
    if make_sample_weight:
        sample_weight = make_sample_weight(X, y, **kwargs)
    else:
        sample_weight = None
    return X, y, sample_weight


def _coords_from_other_types(coords):
    if isinstance(coords, dict):
        return coords.__getitem__
    if (isinstance(coords, Sequence) and all(isinstance(c, Sequence) for c in coords)):
        coords = dict(coords)
        return coords.__getitem__
    return coords.__getattr__



def _X_to_type(item, typ, dims, shape):
    if typ == 'xarray:Dataset':
        X_data = OrderedDict()
        coords = OrderedDict(tuple((d, np.arange(s)) for d, s in zip(dims, shape)))
        for j in range(item.shape[1]):
            arr = xr.DataArray(item[:, j].reshape(shape),
                               coords=coords,
                               dims=dims)
            X_data['band_' + str(j)] = arr
        return xr.Dataset(X_data)

def _y_to_type(*args, **kwargs):
    return args[0]

_sw_to_type = _y_to_type

def sklearn_output_as_format(sklearn_out,
                             x_output_type='xarray:Dataset',
                             y_output_type='numpy:ndarray',
                             sample_weight_output_type='numpy:ndarray',
                             shape=None,
                             as_flat=False,
                             dims=None,
                             reshaper=None,
                             make_sample_weight=None,
                             **kwargs):
    """Make dataset based on the output of a sklearn.dataset function.

    Parameters
    ----------
    sklearn_out: output from a sklearn.datasets datagen function
    x_output_type: string (default: 'xarray:Dataset')
    y_output_type: string (default: 'numpy:ndarray')
    sample_weight_output_type: type of sample weights (default: 'numpy:ndarray')
    shape: tuple, optional
        [TODO] Currently required in code. Either require in signature or do
        something else but raise an error when the default None value is
        passed.
    as_flat: boolean, optional
    dims: tuple of strings, optional
    reshaper: type? [TODO] unused in code
    make_sample_weight: function
        See _split_sklearn_out.

    Returns
    -------
    X2: xarray.DataSet of features
    y2: 1-dimensional numpy.array of labels
    sample_weight2: 1-dimensional numpy.array of sample weights, same shape as y2

    """
    if not shape:
        raise ValueError('Expected shape keyword argument')

    typs = (x_output_type, y_output_type, sample_weight_output_type)
    x_typ, y_typ, sw_output_type = (_find_type(t) for t in typs)
    X, y, sample_weight = _split_sklearn_out(sklearn_out,
                                             make_sample_weight=make_sample_weight,
                                             x_output_type=x_output_type,
                                             y_output_type=y_output_type,
                                             dims=dims,
                                             shape=shape,
                                             as_flat=as_flat,
                                             reshaper=reshaper,
                                             **kwargs)
    X2 = _X_to_type(X, x_typ, dims, shape)
    y2 = _y_to_type(y, y_typ, dims, shape)
    sample_weight2 = _sw_to_type(sample_weight, sw_output_type, dims, shape)
    return X2, y2, sample_weight2


def make_base(sklearn_func, *args, **kwargs):
    """Make dataset based on sklearn datagen function.

    Parameters    
    ----------
    sklearn_func : function from sklearn.datasets
        Examples: make_blobs, make_regression, etc.
    args : positional arguments passed to sklearn_func
    kwargs : keyword arguments passed to sklearn_func and sklearn_output_as_format
        You may also pass a 'shape' (tuple of integers) and corresponding
        dimension names 'dims' (tuple of strings, same size as 'shape'). This
        follows the xarray.DataArray conventions. 

    Returns
    -------
    X: xarray.DataSet of features
    y: 1-dimensional numpy.array of labels
    sample_weight: 1-dimensional numpy-array of sample weights, same shape as y
        See documentation of [TODO] for this.

    See Also
    --------
    sklearn_output_as_format
    """
    if not kwargs.get('shape'):
        kwargs['shape'] = DEFAULT_SHAPE
    kw = {k: v for k,v in kwargs.items() if k not in NON_SKLEARN_KWARGS}
    kw['n_samples'] = np.prod(kwargs['shape'])
    sklearn_out = sklearn_func(*args, **kw)
    if not kwargs.get('dims'):
        kwargs['dims'] = tuple('xyzt')[:len(kwargs['shape'])]
    X, y, sample_weight = sklearn_output_as_format(sklearn_out, **kwargs)
    return X, y, sample_weight


make_regression = partial(make_base, skdatasets.make_regression)
make_blobs = partial(make_base, skdatasets.make_blobs)
