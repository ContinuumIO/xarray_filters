#!/bin/bash

set -e

test -z "$ANACONDA_UPLOAD_USER" && echo "Please set the ANACONDA_UPLOAD_USER environment variable" && exit 1
test -z "$ANACONDA_UPLOAD_TOKEN" && echo "Please set the ANACONDA_UPLOAD_TOKEN environment variable" && exit 1
test -z "$CHANNELS" && echo "Please set the CHANNELS environment variable" && exit 1
test -z "$PYTHON" && echo "Please set the PYTHON environment variable" && exit 1
test -z "$NUMPY" && echo "Please set the NUMPY environment variable" && exit 1

set -x

conda install --name root anaconda-client
pkg_fpath=~/miniconda/conda-bld/osx-64/xarray_filters-0.0.1-py36_18.tar.bz2 #`conda build $CHANNELS --output --python $PYTHON --numpy $NUMPY conda.recipe | sed 's/-py[^_]*/-py\*/' | xargs \ls -1`
conda convert -p all -o _pkgs "$pkg_fpath"
find _pkgs -type f -name "*.tar.bz2" -exec \
     anaconda --token "$ANACONDA_UPLOAD_TOKEN" upload --user "$ANACONDA_UPLOAD_USER" --label dev {} \+
