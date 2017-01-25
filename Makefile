PYTHON_MODULES := pyndl
PYTHON_VERSION := 3
PYTHONPATH := .
VENV := .venv
PYTEST := env PYTHONPATH=$(PYTHONPATH) PYTEST=1 $(VENV)/bin/py.test
PEP8 := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/pep8 --repeat
PYTHON := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python$(PYTHON_VERSION)
PIP := $(VENV)/bin/pip

DEFAULT_PYTHON := /usr/bin/python$(PYTHON_VERSION)
VIRTUALENV := /usr/bin/env virtualenv

REQUIREMENTS := -r requirements.txt

default: checkstyle test

use-venv:
		bash -c 'source $(VENV)/bin/activate'
install-venv:
		$(VIRTUALENV) -p $(DEFAULT_PYTHON) -q $(VENV)
install: install-venv
		$(PIP) install -q $(REQUIREMENTS)
checkstyle: use-venv
		$(PEP8) $(PYTHON_MODULES)
test: use-venv
		$(PYTEST) $(PYTHON_MODULES)
test-slow: use-venv
		$(PYTEST) --runslow $(PYTHON_MODULES)
.PHONY: default install test use-venv install-venv checkstyle test-slow
