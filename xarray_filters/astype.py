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



