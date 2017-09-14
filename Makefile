.PHONY: clean, all, docs, test, install, test_py35.log, html_cov_py_35, test_py27.log, html_cov_py_27

clean:
	rm -rf test_py*.log html_cov_py*

install:
	conda env create -f environment.yml -n xrf_testenv

docs:
	cd docs && make html && cd ..

test:
	pytest

