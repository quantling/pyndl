PYTHON_MODULES := pyndl tests
PYTHON_VERSION := ''
PYTHONPATH := .
VENV := .venv
PYTEST := env PYTHONPATH=$(PYTHONPATH) PYTEST=1 $(VENV)/bin/py.test
PEP8 := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/pep8 --repeat
PYTHON := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python$(PYTHON_VERSION)
PIP := $(VENV)/bin/pip

DEFAULT_PYTHON := /usr/bin/python$(PYTHON_VERSION)
VIRTUALENV := /usr/bin/env virtualenv


default:
		tox
install: install-venv
		which tox > /dev/null || (echo "Please install tox (pip install tox)!" && exit 1) && echo "All right! Run with tox."
checkstyle:
		tox -e checkstyle
docs:
		tox -e docs
test:
		tox -e test
test-versions:
		tox -e $(tox -l | grep test | paste -d, -s)
test-slow: use-venv
		tox -e test -- --runslow
.PHONY: default install test use-venv install-venv checkstyle test-slow
