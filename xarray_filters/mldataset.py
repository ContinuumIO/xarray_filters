from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from collections import OrderedDict
from functools import wraps

import pandas as pd
import xarray as xr

from xarray_filters.chain import chain
from xarray_filters.reshape import (to_features,
                                    from_features,
                                    concat_ml_features,
                                    has_features)
from xarray_filters.constants import FEATURES_LAYER, YNAME

__all__ = ['MLDataset',]


class MLDataset(xr.Dataset):
    '''Wraps xarray.Dataset for chainable preprocessors and
    reshaping to feature matricies that may be inputs to
    scikit-learn or similar models

    TODO - Gui: doctest MLDataset where needed in this
    class defintion - wrap the documentation for to_features,
    from_features, and concat_ml_features, and chain methods
    '''

    def to_features(self,*args, **kwargs):
        '''* TODO Gui - wrap docstring for to_features'''
        return to_features(self, *args, **kwargs)

    def from_features(self, features_layer=None):
        '''* TODO wrap docstring for from_features'''
        if features_layer is None:
            features_layer = FEATURES_LAYER
        if not features_layer in self.data_vars:
            raise ValueError('features_layer ({}) not in self.data_vars'.format(features_layer))
        data_arr = self[features_layer]
        dset = from_features(data_arr)
        return dset

    def chain(self, func_args_kwargs, layers=None):
        '''
        '''
        return chain(self, func_args_kwargs, layers=layers)

    def concat_ml_features(self, *dsets, **kwargs):
        '''TODO - wrap docstring of concat_ml_features'''
        features_layer = kwargs.get('features_layer', FEATURES_LAYER)
        concat_dim = kwargs.get('concat_dim', None)
        keep_attrs = kwargs.get('keep_attrs', False)
        dsets = (self,) + tuple(dsets)
        return concat_ml_features(*dsets,
                                  features_layer=features_layer,
                                  concat_dim=concat_dim,
                                  keep_attrs=keep_attrs)

    def has_features(self, *args, **kwargs):
        return has_features(self, *args, **kwargs)

    def _guess_yname(self, features_layer=None):
        if yname is None:
            yname = YNAME
        return yname

    def _extract_y_from_features(self, dset=None, y=None, features_layer=None,
                                 yname=YNAME, y1d=True, as_np=True):
        features_layer = features_layer or FEATURES_LAYER
        features_layer = has_features(dset, features_layer=features_layer, raise_err=False)
        if dset is None:
            dset = self
        if features_layer:
            arr = dset[features_layer]
            col_dim = arr.dims[1]
            col_labels = getattr(arr, col_dim)
            idxes = [col for col, item in enumerate(col_labels)
                     if col != yname]
            xkw = {col_dim: idxes}
            ykw = {col_dim: yname}
            X = arr.isel(**xkw)
            if y is None:
                if yname in getattr(arr, arr.dims[-1], pd.Series([]).values):
                    y = arr.isel(**ykw)
                    if as_np:
                        y = y.values
            if as_np and y1d and y is not None and y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze()
            if as_np:
                X = X.values
                return X, y
            else:
                return X.to_dataframe(), y.to_dataframe()
        else:
            raise ValueError('TODO --- msg? {}'.format((self, dset)))

    def to_array(self, y=None, features_layer=None, **kw):
        "Return X, y NumPy arrays with given shape"
        features_layer = self.has_features(raise_err=False, features_layer=features_layer)
        if not features_layer:
            dset = self.to_features(features_layer=features_layer, **kw)
        else:
            dset = self
        return self._extract_y_from_features(dset=dset, y=y,
                                             features_layer=features_layer,
                                             as_np=True)


    def to_dataframe(self, layers=None, yname=None):
        "Return a dataframe with features/labels optionally named."
        features_layer = self.has_features(raise_err=False)
        df[yname] = self.y
        return df

    def to_dataset(self, *args, **kw):
        return xr.Dataset(self)

    def to_mldataset(self, *args, **kw):
        return self

    #def astype(self, to_type, **kw):
     #   from xarray_filters.astype import astype
      #  return astype(self, to_type=to_type, **kw)

