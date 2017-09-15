SRCDIR = xarray_filters

.PHONY: clean, all, docs, test,  test_py35.log, html_cov_py_35, test_py27.log, html_cov_py_27

clean:
	rm -rf test_logs .coverage tags
ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	ctags --python-kinds=-i -R $(SRCDIR)

install:
	conda env create -f environment.yml -n xrf_testenv

docs:
	#TODO
	
test:
	pytest

test_all_envs:
	python run_tests.py

develop:
	pip install -e .

