"""
Utility module for handling Python 2 and Python 3 in a single codebase.

Guidelines:
    - Keep it as compatible as possible with the `six` [1]  module. For
      example, six.PY2 == True iff the python major version is 2.
    - Ensure importing all (*) from this module returns only the names you want
      exported (take measures to hide all unnecessary names defined here).
    - If it looks like we will just rewrite six, use six instead.
    - Consider using the `python-future` [2] package.


[1] https://pypi.python.org/pypi/six
[2] https://pypi.python.org/pypi/future

Note: inspired on xarray/core/pycompat.py, xarray version 0.9.6.
"""

from __future__ import division, print_function, unicode_literals
import sys

PYVERSION = (sys.version_info.major, sys.version_info.minor) 
PY2 = sys.version_info.major == 2
PY3 = sys.version_info.major == 3
if PY2:
    assert  PYVERSION >= (2, 7),  'Minimum valid Python 2 version is 2.7'
elif PY3:
    assert  PYVERSION >= (3, 5),  'Minimum valid Python 3 version is 3.5'
else:
    raise RuntimeError('The only suppored versions of Python are: 2.7+, 3.5+')


if PY3:  # pragma: no cover
    basestring = str
    unicode_type = str
    bytes_type = bytes

    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    range = range
    zip, map, filter = zip, map, filter
    from functools import reduce
    import builtins
    from urllib.request import urlretrieve
if PY2:  # pragma: no cover
    basestring = basestring
    unicode_type = unicode
    bytes_type = str

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    range = xrange
    from itertools import izip as zip, imap as map, ifilter as filter
    reduce = reduce
    import __builtin__ as builtins
    from urllib import urlretrieve
