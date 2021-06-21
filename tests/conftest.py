'''
Configuration for py.test-3.

'''

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")
    parser.addoption("--no-linux", action="store_true",
                     help="run without linux tests")


def pytest_runtest_setup(item):
    if 'runslow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run this test")

    if 'nolinux' in item.keywords and item.config.getoption("--no-linux"):
        pytest.skip("run without --no-linux option to run this test")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", 'runslow: marks tests which run very slow (deselect with -m "not runslow")'
    )
    config.addinivalue_line(
        "markers", 'nolinux: marks tests which just run on linux (deselect with -m "not nolinux")'
    )
