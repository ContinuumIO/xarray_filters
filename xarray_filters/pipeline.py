from __future__ import absolute_import, division, print_function, unicode_literals

import types
from collections import OrderedDict

import sklearn
import sklearn.pipeline
import six


class PatchInitSig(type):
    def __new__(cls, name, bases, attr):
        # Create instance variables with the same names as class variables.
        # The class variables serve as defaults, but the instance variables
        # can be modified independently among instance objects.
        setattr_section_strs = []
        params_to_keep = []
        for param, val in attr.items():
            if param not in ('transform','fit', 'fit_transform'):
                if not param.startswith('_'):
                    setattr_section_strs.append('self.{0} = {0}'.format(param))
                    params_to_keep.append(param)
        if not params_to_keep:
            def init_method_noargs(self):
                self.fit_transform = self.transform
            attr['__init__'] = init_method_noargs
        else:
            init_method_str = '''def init_method_withargs(self, {}):
    {}
    self.fit_transform = self.transform
'''.format(', '.join(['{}={}'.format(p, repr(attr[p])) for p in params_to_keep]),
           '\n    '.join(setattr_section_strs))
            #print(init_method_str)
            exec(init_method_str) in globals(), locals()
            attr['__init__'] = locals()['init_method_withargs']
        return super(PatchInitSig, cls).__new__(cls, name, bases, attr)


class Step(six.with_metaclass(PatchInitSig,
                              sklearn.base.BaseEstimator,
                              sklearn.base.TransformerMixin)):
    """As an abstract base class, this should never be instantiated directly.
    Instead, subclasses should be created which implement transform().
    Additionally, __init__ should not be overridden.
    """
    def transform(self, X, y=None, **params):
        """This method must be overridden by subclasses of Step."""



class Generic(Step):
    func = None
    kw = None

    def transform(self, X, y=None, **params):
        p1 = self.get_params()
        func = p1.pop('func', self.func)
        kw = p1.pop('kw', {}) or {}
        kw.update(params)
        kw.update(p1)
        if not func:
            raise ValueError('TODO message')
        return func(X, y=y, **kw)

    def fit(self, X, y=None, **params):
        return self.transform(X, y=y, **params)

    def fit_transform(self, X, y=None, **params):
        return self.transform(X, y=y, **params)


class WriteNetCDF(Step):
    fname = ''
    def transform(self, dset):
        params = self.get_params()

        # Glean the fname from the parameters,
        # defaulting to the constructor's default value.
        fname = params.get('fname', self.fname)
        del params['fname']

        dset.to_netcdf(fname, **params)
        return dset


class Pipeline(sklearn.pipeline.Pipeline):
    def __init__(self, steps, memory=None):
        steps_copy = list(steps)
        if steps_copy[-1][1] is not None:
            steps_copy.append(('iden', None))
        super(Pipeline, self).__init__(steps_copy, memory)

    def _transform(self, X):
        Xt = X
        for name, step in self.steps:
            if step is not None:
                Xt = step.transform(Xt)
        return Xt

    def _inverse_transform(self, X):
        Xt = X
        for name, step in self.steps[::-1]:
            if step is not None:
                Xt = step.inverse_transform(Xt)
        return Xt
