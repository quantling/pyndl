#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

from .. import corpus

def test_read_clean_gzfile():
    file_name = 'tests/resources/xml_gz_corpus/2005/9636/4793488_1of1.xml.gz'
    lines = list(corpus.read_clean_gzfile(file_name, break_duration=5.0))
    assert len(lines) == 1245


def test_main():
    corpus_file = 'tests/temp/corpus.txt'
    corpus.main('tests/resources/xml_gz_corpus', corpus_file, n_threads=1,
            verbose=False)
    os.remove(corpus_file)
    corpus.main('tests/resources/xml_gz_corpus', corpus_file, n_threads=2,
            verbose=True)
    with open(corpus_file, "rt") as new_file:
        lines_new = new_file.readlines()
    with open("./tests/reference/corpus.txt", "rt") as reference:
        lines_reference = reference.readlines()
    assert lines_new == lines_reference
    os.remove(corpus_file)

