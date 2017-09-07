'''TODO - modify this module/function as needed
    to be consistent with similar logic from
    PR 2.  Consider whether we need extra keywords
    for different situations, e.g. to control the
    acceptable return values
    '''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pandas as pd
import xarray as xr

from xarray_filters.constants import YNAME
from xarray_filters.utils import _infer_coords_and_dims


__all__ = ['NpXyTransformer',]



def astype(obj, to_type, **kwargs):
    """Convert to given type.

    self.astype(f, **kwargs) calls self.to_f(**kwargs)

    Valid types are in NpXyTransformer.accepted_types.

    See Also
    --------

    NpXyTransformer.to_dataset
    NpXyTransformer.to_array
    NpXyTransformer.to_dataframe
    NpXyTransformer.to_*
    MLDataset.to_dataset
    MLDataset.to_array
    MLDataset.to_dataframe
    MLDataset.to_*
    etc...
    """
    assert to_type in obj.__class__.accepted_types
    to_method_name = 'to_' + to_type
    to_method = obj.__getattribute__(to_method_name)
    return to_method(**kwargs)


class TypeTransformerBase:
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

    def astype(self, to_type=None, **kw):
        if to_type is None:
            return self
        return astype(self, to_type=to_type, **kw)

class NpXyTransformer(TypeTransformerBase):


    def to_array(self, xshape=None):
        "Return X, y NumPy arrays with given shape"
        if xshape:
            X, y = self.X.reshape(xshape), self.y
        else:
            X, y = self.X, self.y
        return X, y

    def to_dataframe(self, layers=None, yname=None):
        "Return a dataframe with features/labels optionally named."
        df = pd.DataFrame(self.X, columns=layers)
        df[yname] = self.y
        return df

    def to_dataset(self, coords=None, dims=None, attrs=None, shape=None, layers=None, yname=None):
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
        layers : sequence of str, optional
            Name given to each feature (column in self.X)
        yname : str, optional
            Name given to the label variable (self.y).

        Returns
        -------
        dataset = xarra.Dataset
            Each feature (column of self.X) and the label (self.y) becomes a
            data variable in this dataset.


        """
        # Obtain coordinates, dimensions, shape, variable names, etc.
        if shape is None:
            shape = (self.X.shape[0],)
        new_coords, new_dims = _infer_coords_and_dims(shape, coords, dims)
        nvars = self.X.shape[1]
        if not layers:
            layers = ['X' + str(n) for n in range(nvars)]
        # store features X in dataset
        ds = xr.Dataset(attrs=attrs)
        for (xname, col) in zip(layers, self.X.T):
            ds[xname] = xr.DataArray(data=col.reshape(shape),
                                     coords=new_coords,
                                     dims=new_dims)
        # store label y
        if not yname:
            yname = YNAME
        self.y.resize(shape)
        ds[yname] = xr.DataArray(data=self.y,
                                 coords=new_coords,
                                 dims=new_dims)
        return ds

    def to_mldataset(self, coords=None, dims=None, attrs=None, shape=None, layers=None, yname=None):
        '''TODO docs as above ^^'''
        dset = self.to_dataset(coords=coords,
                               dims=dims,
                               attrs=attrs,
                               shape=shape,
                               layers=layers,
                               yname=yname)
        return MLDataset(dset)



