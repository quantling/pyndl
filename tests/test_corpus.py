#!/usr/bin/env python3

# pylint: disable=C0111

import os

from pyndl import corpus

TEST_ROOT = os.path.dirname(__file__)


def test_read_clean_gzfile():
    file_name = os.path.join(TEST_ROOT, 'resources/xml_gz_corpus/2017/0001/1.xml.gz')

    lines = list(corpus.read_clean_gzfile(file_name, break_duration=5.0))
    assert len(lines) == 3


def test_create_corpus_from_gz():
    corpus_file = os.path.join(TEST_ROOT, 'temp/xml_gz_corpus.txt')
    resource_file = os.path.join(TEST_ROOT, 'resources/xml_gz_corpus')
    reference_file = os.path.join(TEST_ROOT, 'reference/xml_gz_corpus.txt')
    corpus.create_corpus_from_gz(resource_file, corpus_file, n_threads=1, verbose=False)
    os.remove(corpus_file)
    corpus.create_corpus_from_gz(resource_file, corpus_file, n_threads=2, verbose=True)
    with open(corpus_file, "rt") as new_file:
        lines_new = new_file.readlines()
    with open(reference_file, "rt") as reference:
        lines_reference = reference.readlines()
    assert lines_new == lines_reference
    os.remove(corpus_file)
