'''
----------------------------

``elm.config.func_signatures``

TODO: add doctests - confirm docstrings adequate

TODO: check Py 2 and 3 compat - see commented code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import inspect
import sys
from collections import OrderedDict

def get_args_kwargs_defaults(func):
    '''Get the required args, defaults, and var keywords of func

    Parameters
    ----------

    func: callable

    Returns
    -------

    tuple of 3 elements:
        args: Positional argument names
        kwargs: Keyword arguments and their defaults
        takes_var_keywords: True if func takes variable keywords

    Examples
    --------

    TODO
    '''
    if hasattr(inspect, 'signature'):
        sig = inspect.signature(func) # Python 3
        empty = inspect._empty
    else:
        import funcsigs
        sig = funcsigs.signature(func) # Python 2
        empty = funcsigs._empty
    params = sig.parameters
    kwargs = {}
    args = []
    takes_variable_keywords = None
    for k, v in params.items():
        if v.default != empty:
            kwargs[k] = v.default
        else:
            args.append(k)
        if v.kind == 4:
            #<_ParameterKind.VAR_KEYWORD: 4>
            takes_variable_keywords = k

        '''sig = inspect.getargpsec(func) # Python 2
        args = sig.args
        kwargs = sig.keywords
        called = None
        for x in range(100):
            test_args = (func,) + tuple(range(x))
            try:
                called = inspect.getcallargs(*test_args)
                break
            except:
                pass
        if called is None:
            raise
        '''
    return args, kwargs, takes_variable_keywords


def filter_args_kwargs(func, *args, **kwargs):
    '''Filter args and kwargs to func based on function
    signature.  If function takes variable number of
    keyword arguments, then do not filter.  In either case,
    ensure args is as long as named positional arguments to
    func if named arguments are not found in kwargs

    Returns
    -------

    args_kw: dict containing required positional and keyword arguments

    Examples
    --------

    TODO
    '''
    kw = kwargs.copy()
    arg_spec, kwarg_spec, takes_variable_keywords = get_args_kwargs_defaults(func)
    args_kw = {}
    for idx, name in enumerate(arg_spec):
        if idx < len(args):
            args_kw[name] = args[idx]
        if name in kw:
            args_kw[name] = kw[name]
    for k, v in kw.items():
        if k in kwarg_spec or takes_variable_keywords:
            args_kw[k] = v
    for k, v in kwarg_spec.items():
        if k not in args_kw:
            args_kw[k] = v
    return args_kw



def filter_kw_and_run_init(__init_method, **kw):
    ___d = OrderedDict(locals())
    kw = filter_args_kwargs(__init_method, **kw)
    __init_method(**kw)


__all__ = ['get_args_kwargs_defaults', 'filter_args_kwargs', 'filter_kw_and_run_init']
