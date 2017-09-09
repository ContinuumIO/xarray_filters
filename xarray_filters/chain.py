from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from collections import OrderedDict

from xarray_filters.pipe_utils import for_each_array

__all__ = ['chain',]

def format_chain_args(trans):
    '''TODO - Document the list of list structures
    that can be passed transforms.  See
    comments in tests/test_reshape.py for now

    Parameters:
        :trans: "transforms" arguments to MLDataset.chain

    Returns:
        list of lists like [[func1, args1, kwargs1],
                            [func2, args2, kwargs2]]
        to be passed to MLDataset.pipe(func, *args, **kwargs) on
        each func, args, kwargs.  Note
    '''
    output = []
    if callable(trans):
        output.append([trans, [], {}])
    elif isinstance(trans, (tuple, list)):
        trans = list(trans)
        for tran in trans:
            if callable(tran):
                output.append([tran, [], {}])
            elif isinstance(tran, (list, tuple)) and tran:
                tran = list(tran)
                if isinstance(tran[-1], dict):
                    kw = tran[-1]
                    kw_idx = len(tran) - 1
                else:
                    kw = dict()
                    kw_idx = None
                func = [idx for idx, _ in enumerate(tran)
                        if isinstance(_, str) or callable(_)]
                if not func:
                    raise ValueError('Expected a string DataArray method name or a callable in {}'.format(tran))
                args = [_ for idx, _ in enumerate(tran)
                        if idx != kw_idx and idx not in func]
                func = tran[func[0]]
                tran = [func, args, kw]
                output.append(tran)
    return output


def chain(dset, func_args_kwargs, layers=None):
    from xarray_filters.mldataset import MLDataset
    func_args_kwargs = format_chain_args(func_args_kwargs)

    if layers is not None:
        if any(layer not in dset.data_vars for layer in layers):
            raise ValueError('At least one of layers - {} - is not in data_vars: '.format(layers, dset.data_vars))
        arrs = (dset[layer] for layer in layers
                if layer in dset.data_vars)
    else:
        layers, arrs = dset.data_vars.keys(), dset.data_vars.values()
    new_dset = MLDataset(OrderedDict(zip(layers, arrs)))

    for func, args, kwargs in func_args_kwargs:
        if not callable(func):
            func = for_each_array(func)
        dset = dset.pipe(func, *args, **kwargs)
    return dset


