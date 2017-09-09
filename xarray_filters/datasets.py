"""
Overview
--------

This module contains a routines to convert the output of various
`sklearn.datasets.make_*` functions, like

    sklearn.datasets.make_blobs
    sklearn.datasets.make_regression
    sklearn.datasets.make_classification

to the following containers

    xarray.Dataset
    numpy.array
    pandas.DataFrame

as well as allowing us to do some simple postprocessing of the data
conveniently, like reshaping naming variables, dimensions, etc.

The idea is to provide drop-in replacements for the `sklearn.make_*` functions
but with additional features, especially around producing multidimensional
data.


Why this is useful
------------------

In natural sciences higher-dimensional data is common, and using labeled
dimensions and coordinates (tick labels) makes it much easy to work with the
data.

In other words, we would like the data simulation capabilities from
sklearn.datasets, but returning xarray.Dataset objects. While we are at it, we
might as well support other data structures too, like pandas.DataFrame.

Motivating usecases:

- Generate data to test models that predict temperature or humidity at
  3d-spatial coordinates and time based on earlier measurements at nearby/same
  coordinates.


Implementation Notes
--------------------

We create new data simulation functions based on the original sklearn ones.
Each one of the new functions has the same signature as in sklearn, as well as
some additional, optional keyword-only arguments:

- `astype` to cast the data to different types (defaults to `xarray.Dataset`)
- `**kwargs` with optional keyword arguments that depend on what is passed to
  `astype`.

For example, to obtain the exact same behavior as in sklearn, you can pass
`astype='array'`:

To support all the conversions, we create a class `NpXyTransformer` that has one
method (`to_array`, `to_dataset`, `to_dataframe`, etc.) per conversion. In
addition, we implement a `_make_base` function that maps a
`sklearn.datasets.make_*` function to the new, extended version, with
appropriate docstring and signature.

In a nutshell, the higher level API is like

>>> m = make_blobs(n_samples=5, n_features=2,  # sklearn args
...     astype='dataset', xnames=['feat_1', 'feat_2'])  # new args

At a lower level, that is equivalent to

>>> make_blobs = _make_base(sklearn.datasets.make_blobs)
>>> transformer = make_blobs(astype=None)  # this is a NpXyTransformer object
>>> m = transformer.to_dataset(xnames=['feat_1', 'feat_2'])

The full list of converted functions from sklearn is in converted_make_funcs:
>>> sorted(converted_make_funcs.keys())  # doctest: +NORMALIZE_WHITESPACE
['make_blobs',
 'make_circles',
 'make_classification',
 'make_friedman1',
 'make_friedman2',
 'make_friedman3',
 'make_gaussian_quantiles',
 'make_hastie_10_2',
 'make_low_rank_matrix',
 'make_moons',
 'make_multilabel_classification',
 'make_regression',
 'make_s_curve',
 'make_sparse_spd_matrix',
 'make_sparse_uncorrelated',
 'make_swiss_roll']

The full list of types that can be used for conversion (i.e. can be passed to
the keyword `astype`) is
>>> NpXyTransformer.accepted_types
('array', 'dataframe', 'dataset', 'mldataset')


"""



import inspect
import string

import numpy as np
import xarray as xr
import pandas as pd
import sklearn.datasets
import logging

from collections import Sequence, OrderedDict, defaultdict
from functools import partial, wraps

from . utils import _infer_coords_and_dims
from . mldataset import MLDataset



logging.basicConfig()
logger = logging.getLogger(__name__)


class NpXyTransformer:
    "Transforms a pair (feature_matrix, labels_vector) with to_* methods."
    # Transform methods are to_f where f in accepted types
    accepted_types = ('array', 'dataframe', 'dataset', 'mldataset')
    default_type = 'mldataset'

    def __init__(self, X, y=None):
        """Initalizes an NpXyTransformer object.

        Access the underlying feature matrix X and labels y with self.X and
        self.y, respectively.
        """
        self.X = X  # always a 2d numpy.array
        self.y = y  # always a 1d numpy.array

    def to_array(self, xshape=None):
        """Return X, y NumPy arrays with given shape

        Parameters
        ----------
        xshape: tuple
            Desired shape for the resulting feature array.

        Returns
        -------
        X: numpy.ndarray
            Feature array with shape `xshape` is `xshape` is supplied,
            otherwise the original shape of self.X.
        y: numpy.ndarray
            One dimensional array of labels for the data (the variable we are
            trying to predict).

        Examples
        --------
        >>> transformer = make_blobs(n_samples=4, n_features=3, random_state=0,
        ...     astype=None)
        >>> X, y = transformer.to_array()
        >>> X.shape
        (4, 3)
        >>> X, y = transformer.to_array(xshape=(6, 2))
        >>> X.shape
        (6, 2)
        """
        if xshape:
            X, y = self.X.reshape(xshape), self.y
        else:
            X, y = self.X, self.y
        return X, y

    def to_dataframe(self, xnames=None, yname=None):
        """Return a single dataframe with features, labels optionally named.

        Parameters
        ----------
        xnames: sequence of str
            Feature names.
        yname: str
            Name of the label variable (variable we are trying to predict).

        Returns
        -------
        df: pandas.DataFrame
            One column per feature, the last column is the label.

        Examples
        --------
        >>> transformer = make_regression(n_samples=5, n_features=2, random_state=0,
        ...     astype=None)
        >>> df = transformer.to_dataframe(xnames=['temp', 'pressure'], yname='humidity')
        >>> type(df)
        <class 'pandas.core.frame.DataFrame'>
        >>> df.columns
        Index(['temp', 'pressure', 'humidity'], dtype='object')
        """
        nfeatures = self.X.shape[1]
        if not xnames:
            xnames = ['X' + str(n) for n in range(nfeatures)]
        if not yname:
            yname = 'y'
        df = pd.DataFrame(self.X, columns=xnames)
        df[yname] = self.y
        return df

    def to_dataset(self, coords=None, dims=None, attrs=None, shape=None, xnames=None, yname=None):
        """Return an xarray.DataSet with given shape, coords/dims/var names.

        Parameters
        ----------
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates.
        dims : str or sequence of str, optional
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.
        shape: tuuple, optional
            Length of each dimension, or equivalently, number of elements in each
            coordinate.
        xnames : sequence of str, optional
            Name given to each feature (column in self.X).
        yname : str, optional
            Name given to the label variable (self.y).

        Returns
        -------
        dataset = xarra.Dataset
            Each feature (column of self.X) and the label (self.y) becomes a
            data variable in this dataset.


        """
        # Obtain coordinates, dimensions, shape, variable names, etc.
        if not shape:
            shape = (self.X.shape[0],)
        new_coords, new_dims = _infer_coords_and_dims(shape, coords, dims)
        nfeatures = self.X.shape[1]
        if not xnames:
            xnames = ['X' + str(n) for n in range(nfeatures)]
        # store features X in dataset
        ds = xr.Dataset(attrs=attrs)
        for (xname, col) in zip(xnames, self.X.T):
            ds[xname] = xr.DataArray(data=col.reshape(shape), coords=new_coords, dims=new_dims)
        # store label y
        if not yname:
            yname = 'y'
        ds[yname] = xr.DataArray(data=self.y.reshape(shape), coords=new_coords, dims=new_dims)
        return ds

    def to_mldataset(self, coords=None, dims=None, attrs=None, shape=None, layers=None, yname=None):
        '''TODO docs as above ^^'''
        dset = self.to_dataset(coords=coords,
                               dims=dims,
                               attrs=attrs,
                               shape=shape,
                               xnames=layers,
                               yname=yname)
        return MLDataset(dset)

    def astype(self, to_type, **kwargs):
        """Convert to given type.

        self.astype(f, **kwargs) calls self.to_f(**kwargs)

        Valid types are in NpXyTransformer.accepted_types.

        See Also
        --------

        NpXyTransformer.to_dataset
        NpXyTransformer.to_array
        NpXyTransformer.to_dataframe
        NpXyTransformer.to_*
        etc...
        """
        assert to_type in self.__class__.accepted_types
        to_method_name = 'to_' + to_type
        to_method = self.__getattribute__(to_method_name)
        return to_method(**kwargs)


def _make_base(skl_sampler_func):
    """Maps a make_* function from sklearn.datasets to an extension.

    The extended version of the function implements transformations through the
    various `NpXyTransformer.to_*` methods, enabled in the new function via the
    `astype` keyword and additional, optional keyword arguments.

    The goal is to use the make_* functions from sklearn to generate the data,
    but then postprocess it with the various to_* methods from a NpXyTransformer
    class.

    Parameters
    ----------
    skl_sampler_func: a make_* function from sklearn.datasets

    Returns
    -------
    wrapper: a function that extends skl_sampler_func

    Notes
    -----
    - The signature of the new function is the same signature as the original
      sklearn function with the addition of some keyword-only arguments. This
      facilitates introspection and is useful when calling help() on the new
      function.
    - The docstring for the new function is automatically generated to provide all
      the information the user needs to use the infuction: what can be passed to
      `astype`, where to find the valid keywords for a given type: in
      `NpXyTransformer.to_dataset` for `astype='dataset'`, etc.
    - A decorator doesn't solve this. It could add the functionality, but it
      would not change the docstring or the signature of the function, as it
      would keep the original one from the wrapped function from sklearn. So we
      have to do the process manually.

    Examples
    --------
    The functions we create generate the same data as the corresponding sklearn
    functions.

    >>> make_classification = _make_base(sklearn.datasets.make_classification)
    >>> Xskl, yskl = sklearn.datasets.make_classification(random_state=0)
    >>> our_data = make_classification(random_state=0)
    >>> np.allclose(Xskl[:, 0], our_data['X0'])  # comparing floats
    True
    >>> np.allclose(Xskl[:, 1], our_data['X1'])  # comparing floats
    True

    Same goes for other features, as well as the label

    >>> np.all(np.equal(yskl, our_data['y']))  # comparing ints
    True

    The signature of each wrapper is identical to its sklearn equivalent, with
    the addition of the `astype` keyword and additional, optional keywords to
    go with `astype`

    >>> inspect.signature(sklearn.datasets.make_classification)  # doctest: +NORMALIZE_WHITESPACE
    <Signature (n_samples=100, n_features=20, n_informative=2, n_redundant=2,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
    flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
    shuffle=True, random_state=None)>
    >>> inspect.signature(make_classification)  # doctest: +NORMALIZE_WHITESPACE
    <Signature (n_samples=100, n_features=20, n_informative=2, n_redundant=2,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
    flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
    shuffle=True, random_state=None, *, astype='dataset', **kwargs)>

    If `astype=None`, then the `make_*` function behaves just like in sklearn, 
    but returns a NpXyTransformer object that has various methods to postprocess that
    (X, y) data. For example, we can generate a dataframe of simulated data for
    a regression exercise with

    >>> df1 = make_regression(n_samples=5, n_features=2, random_state=0,
    ...     astype='dataframe', xnames=['thing1', 'thing2'])

    or, equivalently,

    >>> transformer = make_regression(n_samples=5, n_features=2, random_state=0,
    ...     astype=None)  #  this is a NpXyTransformer object
    >>> df2 = transformer.to_dataframe(xnames=['thing1', 'thing2'])

    Verifying:

    >>> np.allclose(df1, df2)  # comparing floats
    True
    """
    skl_argspec = inspect.getfullargspec(skl_sampler_func)
    # Here is where we use the assumption that the make_* function from sklearn
    # has all positional or keyword arguments, all of them with defaults; it
    # could be easily adapted to more flexible setups. TODO: make this more
    # robust; users of the library may apply this to functions that do not
    # satisfy the assumptions listed above.
    assert not skl_argspec.varargs, "{} has variable positional arguments".format(skl_sampler_func.__name__)
    assert not skl_argspec.kwonlyargs, "{} has keyword-only arguments".format(skl_sampler_func.__name__)
    assert len(skl_argspec.args) == len(skl_argspec.defaults), \
            "Some args of {} have no default value".format(skl_sampler_func.__name__)
    skl_params = [inspect.Parameter(name=pname, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, default=pdefault)
                  for (pname, pdefault) in zip(skl_argspec.args, skl_argspec.defaults)]
    default_astype = 'dataset'
    def wrapper(*args, **kwargs):
        '''
        All optional/custom args are keyword arguments.
        '''
        # Step 1: process positional and keyword arguments
        skl_kwds = skl_argspec.args + skl_argspec.kwonlyargs
        # splitting arguments into disjoint sets; astype is a special argument
        skl_kwargs = {k: val for (k, val) in kwargs.items() if k in skl_kwds}
        astype = kwargs.get('astype', default_astype)
        type_kwargs = {k: val for (k, val) in kwargs.items() if (k not in
            skl_kwds and k != 'astype')} 

        # Step 2: obtain the NpXyTransformer object
        # First we need to check that we can handle the output of skl_sampler_func
        out = skl_sampler_func(*args, **skl_kwargs)
        if len(out) != 2:
            error_msg = 'Function {} must return a tuple of 2 elements'.format(skl_sampler_func.__name__)
            raise ValueError(error_msg)
        else:
            X, y = skl_sampler_func(*args, **skl_kwargs)
            if y.ndim != 1:
                raise ValueError("Y must have dimension 1.")
            if X.shape[0] != y.shape[0]:
                raise ValueError('X and y must have the same number of rows')
        Xyt = NpXyTransformer(X, y)

        # Step 3: convert the data to the desired type
        if astype is None:
            sim_data = Xyt
        else:
            sim_data = Xyt.astype(to_type=astype, **type_kwargs)

        return sim_data

    # We now have some tasks: (1) fix the docstring and (2) fix the signature of `wrapper`.
    # Task 1 of 2: fixing the docstring of `wrapper`
    preamble_doc = """Like {skl_funcname}, but with added functionality.

    Parameters
    ---------------------
    Same parameters/arguments as {skl_funcname}, in addition to the following
    keyword-only arguments:

    astype: str
        One of {accepted_types} or None to return an NpXyTransformer. See documentation
        of NpXyTransformer.astype.
        
    **kwargs: dict
        Optional arguments that depend on astype. See documentation of
        NpXyTransformer.astype. 

    See Also
    --------
    {skl_funcname}
    {xy_transformer}

    """.format(
            skl_funcname=(skl_sampler_func.__module__ + '.' + skl_sampler_func.__name__),
            xy_transformer=(NpXyTransformer.__module__ + '.' + NpXyTransformer.__name__),
            accepted_types=(str(NpXyTransformer.accepted_types))

    )
    wrapper.__doc__ = preamble_doc # + skl_sampler_func.__doc__
    # Task 2 of 2: fixing the signature of `wrapper`
    astype_param = inspect.Parameter(name='astype', kind=inspect.Parameter.KEYWORD_ONLY, default=default_astype)
    kwargs_param = inspect.Parameter(name='kwargs', kind=inspect.Parameter.VAR_KEYWORD)
    params = skl_params + [astype_param, kwargs_param]
    wrapper.__signature__ = inspect.Signature(params)
    wrapper.__name__ = skl_sampler_func.__name__
    return wrapper


def fetch_lfw_people(*args, **kw):
    '''TODO Gui wrap docs from sklearn equivalent
    and add notes about it returning MLDataset

    out = fetch_lfw_people()
    In [6]: out.ariel_sharon_65
Out[6]:
<xarray.DataArray 'ariel_sharon_65' (row: 62, column: 47)>
array([[ 111.666664,  128.333328,  123.333336, ...,  186.333328,  188.      ,
         188.333328],
       [ 103.      ,  124.666664,  121.      , ...,  184.333328,  186.      ,
         187.333328],
       [  97.      ,  119.333336,  114.666664, ...,  178.333328,  180.333328,
         183.333328],
       ...,
       [  23.666666,   24.      ,   20.      , ...,  182.      ,  192.666672,
         200.      ],
       [  21.333334,   20.666666,   18.      , ...,  191.      ,  201.      ,
         202.666672],
       [  20.666666,   18.      ,   14.666667, ...,  197.333328,  202.      ,
         199.333328]], dtype=float32)
Coordinates:
  * row      (row) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ...
  * column   (column) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ...
Attributes:
    id:       373
    name:     Ariel Sharon
    '''
    raise NotImplementedError
    #out = sklearn.datasets.fetch_lfw_people(*args, **kw)
    #dset = OrderedDict()
    #name_tracker = defaultdict(lambda: -1)
    #for img, name_id in zip(out.images, out.target):
    #    name = out.target_names[name_id]
    #    name_tracker[name] += 1
    #    clean_name = name.lower().replace(' ', '_') + '_' + str(name_tracker[name])
    #    r, c = img.shape
    #    coords = OrderedDict([('row', np.arange(r)),
    #                          ('column', np.arange(c))])
    #    dims = ('row', 'column',)
    #    attrs = dict(name=name, id=name_id)
    #    data_arr = xr.DataArray(img, coords=coords, dims=dims, attrs=attrs)
    #    dset[clean_name] = data_arr
    #attrs = dict(metadata=[args, kw])
    ## TODO - PR 3's MLDataset instead of Dataset
    #dset = xr.Dataset(dset, attrs=attrs)
    ## TODO - allow converting to dataframe, numpy array?
    #return dset


# Convert all sklearn functions that admit conversion
converted_make_funcs = dict()  # holds converted sklearn funcs
sklearn_make_funcs = [_ for _ in dir(sklearn.datasets) if _.startswith('make_')]  # conversion candidates
for func_name in sklearn_make_funcs:
    try:
        func = getattr(sklearn.datasets, func_name)
        converted_make_funcs[func_name] = _make_base(func)
    except (ValueError, AssertionError) as e:
        warning_msg = "Cannot convert function {}. ".format(func_name) + str(e)
        logger.info(warning_msg)
globals().update(converted_make_funcs)  # careful with overwrite here


extras = ['NpXyTransformer', 'fetch_lfw_people']
__all__ = list(converted_make_funcs) + extras
