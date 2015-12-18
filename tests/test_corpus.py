#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

from .. import corpus


def test_corpus():
    corpus_file = 'tests/temp/corpus.txt'
    main('tests/resources/xml_gz_corpus', corpus_file, n_threads=1,
            verbose=False)
    os.remove(corpus_file)
    main('tests/resources/xml_gz_corpus', corpus_file, n_threads=2,
            verbose=True)
    os.remove(corpus_file)

