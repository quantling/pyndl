'''
Configuration for py.test-3.

'''
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")


def pytest_runtest_setup(item):
    if 'runslow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run this test")
