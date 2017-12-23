#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda;
  if [[ $TOXENV == "py35"]]; then
    conda install python=3.5;
  elif [[$TOXENV == "py36"]]; then
    conda install python=3.6;
  fi
else
    pip install tox-travis;
fi
