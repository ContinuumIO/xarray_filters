#!/usr/bin/env python3

import os
import sys
import requests
import jinja2

recipe_dpath = './dask-ml'
meta_yml_fpath = os.path.join(recipe_dpath, 'meta.yaml')
if os.path.isfile(meta_yml_fpath):
    print('Recipe exists: skipping download.')
    sys.exit(0)

meta_yml_url = 'https://raw.githubusercontent.com/TomAugspurger/staged-recipes/dask-ml/recipes/dask-ml/meta.yaml'
resp = requests.get(meta_yml_url)
assert resp.status_code == 200, 'Error fetching meta.yaml file'
template = jinja2.Template(resp.text)
os.makedirs(recipe_dpath)
print('Writing '+meta_yml_fpath)
with open(meta_yml_fpath, 'w') as fp:
    fp.write(template.render())
