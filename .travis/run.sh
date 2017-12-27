#!/bin/bash

set -e
set -x

PYVER=`python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'`

# setup OSX
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    if which pyenv > /dev/null; then
        eval "$(pyenv init -)"
    fi
    pyenv activate pyndl
fi

tox
tox -e checkstyle
tox -e testcov
tox -e lint
tox -e checktypes
tox -e docs
