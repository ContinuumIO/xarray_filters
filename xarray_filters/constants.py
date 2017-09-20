from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
__all__ = ['FEATURES_LAYER', 'FEATURES_LAYER_DIMS']

FEATURES_LAYER = 'features'
FEATURES_LAYER_DIMS = ('space', 'layer',)
YNAME = 'y'

DASK_FEATURE_CHUNKS = (1000000, 10)
DASK_CHUNK_N = np.prod(DASK_FEATURE_CHUNKS)