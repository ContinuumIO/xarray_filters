.PHONY: clean, all, docs, test, install

clean:
	conda env remove -n xrf_testenv
	rm -rf *.out *.xml htmlcov

install:
	conda env create -f environment.yml -n xrf_testenv

docs: install
	cd docs && make html && cd ..

test: install
	source activate xrf_testenv && py.test
