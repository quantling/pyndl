#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda;
  export PATH="$HOME/miniconda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda config --add channels conda-forge
  conda info -a
  case "${TOXENV}" in
        py35)
            conda install python=3.5;
            ;;
        py36)
            conda install python=3.6;
            ;;
  esac
  conda info
  which python
else
    pip install tox-travis;
fi
