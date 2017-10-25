#!/usr/bin/env python3

import requests
import jinja2
import os
import subprocess

recipe_dpath = './dask-ml'
if os.path.isdir(recipe_dpath):
    print('Recipe exists: skipping download.')

meta_yml_url = 'https://raw.githubusercontent.com/TomAugspurger/staged-recipes/fea51d28eefc935a620575c5c00cc679f0a79766/recipes/dask-ml/meta.yaml'
resp = requests.get(meta_yml_url)
assert resp.status_code == 200, 'Error fetching meta.yaml file'
template = jinja2.Template(resp.text)
os.makedirs(recipe_dpath)
with open(os.path.join(recipe_dpath, 'meta.yaml'), 'w') as fp:
    fp.write(template.render())
