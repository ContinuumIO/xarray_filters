from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict, defaultdict
from xarray.core.indexing import PandasIndexAdapter
from xarray.core.variable import as_variable
import six


def is_dict_like(value):
    "Note: Copied from xarray.utils.py version 0.9.6"
    return hasattr(value, 'keys') and hasattr(value, '__getitem__')


def assert_unique_multiindex_level_names(variables):
    """Check for uniqueness of MultiIndex level names in all given
    variables.

    Not public API. Used for checking consistency of DataArray and Dataset
    objects.

    Note: original xarray docs above.

    Note: Copied from xarray.variable.py version 0.9.6 as it was not part
    of the xarray public API.
    """
    level_names = defaultdict(list)
    for var_name, var in variables.items():
        if isinstance(var._data, PandasIndexAdapter):
            idx_level_names = var.to_index_variable().level_names
            if idx_level_names is not None:
                for n in idx_level_names:
                    level_names[n].append('%r (%s)' % (n, var_name))

    for k, v in level_names.items():
        if k in variables:
            v.append('(%s)' % k)

    duplicate_names = [v for v in level_names.values() if len(v) > 1]
    if duplicate_names:
        conflict_str = '\n'.join([', '.join(v) for v in duplicate_names])
        raise ValueError('conflicting MultiIndex level name(s):\n%s'
                         % conflict_str)


def _infer_coords_and_dims(shape, coords, dims):
    """All the logic for creating a new DataArray

    Note: Copied with minor modifications from xarray.variable.py version 0.9.6
    as it was not part of the xarray public API.

    """

    if (coords is not None and not is_dict_like(coords) and
                len(coords) != len(shape)):
        raise ValueError('coords is not dict-like, but it has %s items, '
                         'which does not match the %s dimensions of the '
                         'data' % (len(coords), len(shape)))

    if isinstance(dims, six.string_types):
        dims = (dims,)

    if dims is None:
        dims = ['dim_%s' % n for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            # try to infer dimensions from coords
            if is_dict_like(coords):
                raise TypeError('inferring DataArray dimensions from dictionary '
                              'like ``coords`` has been deprecated. Use an '
                              'explicit list of ``dims`` instead.')
            else:
                for n, (dim, coord) in enumerate(zip(dims, coords)):
                    coord = as_variable(coord, name=dims[n]).to_index_variable()
                    dims[n] = coord.name
        dims = tuple(dims)
    else:
        for d in dims:
            if not isinstance(d, six.string_types):
                raise TypeError('dimension %s is not a string' % d)

    new_coords = OrderedDict()

    if is_dict_like(coords):
        for k, v in coords.items():
            new_coords[k] = as_variable(v, name=k)
    elif coords is not None:
        for dim, coord in zip(dims, coords):
            var = as_variable(coord, name=dim)
            var.dims = (dim,)
            new_coords[dim] = var

    sizes = dict(zip(dims, shape))
    for k, v in new_coords.items():
        if any(d not in dims for d in v.dims):
            raise ValueError('coordinate %s has dimensions %s, but these '
                             'are not a subset of the DataArray '
                             'dimensions %s' % (k, v.dims, dims))

        for d, s in zip(v.dims, v.shape):
            if s != sizes[d]:
                raise ValueError('conflicting sizes for dimension %r: '
                                 'length %s on the data but length %s on '
                                 'coordinate %r' % (d, sizes[d], s, k))

    assert_unique_multiindex_level_names(new_coords)

    return new_coords, dims


