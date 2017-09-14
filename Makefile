.PHONY: clean, all, docs, test, install, test_py35.log, html_cov_py_35, test_py27.log, html_cov_py_27

clean:
	rm -rf test_py*.log html_cov_py*

install:
	conda env create -f environment.yml -n xrf_testenv

docs:
	cd docs && make html && cd ..

test:
	pytest

test_py%.log: environment_py%.yml
	(source activate xarray_filters_py$* || (conda env create -f $< -n xarray_filters_py$* && source activate xarray_filters_py$*)) \
		&& (py.test --cov-report term:skip-covered --cov > $@ || echo "Some tests failed. See $@ for a report.") \
		&& source deactivate xarray_filters_py$*
	@echo "The conda environment xarray_filters_py$* was not updated."
	@echo "If you want different packages versions in there, you will need to do that manually."

html_cov_py%: environment_py%.yml
	(source activate xarray_filters_py$* || (conda env create -f $< -n xarray_filters_py$* && source activate xarray_filters_py$*)) \
		&& (py.test --cov-report html:$@ --cov-report term:skip-covered --cov > test_py$*.log || echo "Some tests failed. See test_py$*.log for a report.") \
		&& source deactivate xarray_filters_py$*
	@echo "See $@ for a coverage report."
	@echo "The conda environment xarray_filters_py$* was not updated."
	@echo "If you want different packages versions in there, you will need to do that manually."
	
