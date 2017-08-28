from __future__ import absolute_import, division, print_function, unicode_literals

'''
----------------------------

``elm.config.func_signatures``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import inspect
import sys

def get_args_kwargs_defaults(func):
    '''Get the required args, defaults, and var keywords of func

    Parameters:
        :func: callable
    Returns:
        :(args, kwargs, takes_var_keywords): where args are names /
        of required args, kwargs are keyword args with defaults, and
        takes_var_keywords indicates whether func has a \*\*param
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
    '''Remove keys/values from kwargs if cannot be passed to func'''
    kw = kwargs.copy()
    arg_spec, kwarg_spec, takes_variable_keywords = get_args_kwargs_defaults(func)
    new = {}
    missing_args = 0
    for idx, name in enumerate(arg_spec):
        if len(args) > idx:
            new[name] = args[idx]
        elif name in kw:
            new[name] = kw[name]
        elif takes_variable_keywords:
            pass
        else:
            missing_args += 1
    for k, v in kw.items():
        if k in kwarg_spec or takes_variable_keywords:
            new[k] = v
    return new, missing_args



__all__ = ['get_args_kwargs_defaults', 'filter_args_kwargs']
