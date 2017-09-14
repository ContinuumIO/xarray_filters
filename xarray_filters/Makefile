.PHONY: clean, all, docs, test, install

clean:
	conda env remove -n xarray_filters_temp_test_env
	rm -rf *.out *.xml htmlcov

install_py3:
	conda env create -f environment.yml -n xarray_filters_temp_test_env \
	  && source activate xarray_filters_temp_test_env \
              
docs: install
	cd docs && make html && cd ..

test: install
	source venv/bin/activate && \
py.test --cov=sandman2 tests
