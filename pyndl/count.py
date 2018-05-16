"""
pyndl.count
-----------

*pyndl.count* provides functions in order to count

* words and symbols in a corpus file
* cues and outcomes in an event file

"""
# pylint: disable=redefined-outer-name, invalid-name

from collections import Counter
import gzip
import itertools
import multiprocessing
import sys
from typing import Tuple


def _job_cues_outcomes(event_file_name, start, step, verbose=False):
    """
    Counts cues and outcomes for every ``step`` event starting from
    ``start`` event.

    Returns
    -------
    (nn, cues, outcomes) : (int, collections.Counter, collections.Counter)

    """
    cues = Counter()
    outcomes = Counter()
    nn = -1  # in case the for loop never gets called and 1 gets added in the end
    with gzip.open(event_file_name, 'rt') as dfile:
        # skip header
        dfile.readline()
        for nn, line in enumerate(itertools.islice(dfile, start, None, step)):
            cues_line, outcomes_line = line.split('\t')
            for cue in cues_line.split('_'):
                cues[cue] += 1
            for outcome in outcomes_line.strip().split('_'):
                outcomes[outcome] += 1
            if verbose and nn % 100000 == 0:
                print('.', end='')
                sys.stdout.flush()
    return (nn + 1, cues, outcomes)


def cues_outcomes(event_file_name: str,
                  *, number_of_processes=2, verbose=False) -> Tuple[int, Counter, Counter]:
    """
    Counts cues and outcomes in event_file_name using number_of_processes
    processes.

    Returns
    -------
    (n_events, cues, outcomes) : (int, collections.Counter, collections.Counter)

    """
    with multiprocessing.Pool(number_of_processes) as pool:
        step = number_of_processes
        results = pool.starmap(_job_cues_outcomes,
                               ((event_file_name,
                                 start,
                                 step,
                                 verbose)
                                for start in range(number_of_processes)))
        n_events = 0
        cues = Counter()  # type: Counter
        outcomes = Counter()  # type: Counter
        for nn, cues_process, outcomes_process in results:
            n_events += nn
            cues += cues_process
            outcomes += outcomes_process

    if verbose:
        print('\n...counting done.')

    return n_events, cues, outcomes


def _job_words_symbols(corpus_file_name, start, step, lower_case=False,
                       verbose=False):
    """
    Counts the words and symbols for every ``step`` line starting from
    ``start`` line.

    It is assumed that words are separated by at least one space or by a new
    line character.

    .. note::

        Punctuation characters, brackets and some other characters are stripped
        from the word and are not counted.

    Returns
    -------
    (words, symbols) : (collections.Counter, collections.Counter)

    """
    words = Counter()
    symbols = Counter()
    with open(corpus_file_name, 'r') as dfile:
        for nn, line in enumerate(itertools.islice(dfile, start, None, step)):
            for word in line.split():  # splits the string on all whitespace
                word = word.strip()
                word = word.strip('!?,.:;/"\'()^@*~')
                if lower_case:
                    word = word.lower()
                if not word:
                    continue
                words[word] += 1
                symbols += Counter(word)
            if verbose and nn % 100000 == 0:
                print('.', end='')
                sys.stdout.flush()
    return (words, symbols)


def words_symbols(corpus_file_name: str, *,
                  number_of_processes=2, lower_case=False,
                  verbose=False) -> Tuple[Counter, Counter]:
    """
    Counts words and symbols in corpus_file_name using number_of_processes
    processes.

    Returns
    -------
    (words, symbols) : (collections.Counter, collections.Counter)

    """
    with multiprocessing.Pool(number_of_processes) as pool:
        step = number_of_processes
        results = pool.starmap(_job_words_symbols, ((corpus_file_name,
                                                     start,
                                                     step,
                                                     lower_case,
                                                     verbose)
                                                    for start in
                                                    range(number_of_processes)))
        words: Counter = Counter()
        symbols: Counter = Counter()
        for words_process, symbols_process in results:
            words += words_process
            symbols += symbols_process

    if verbose:
        print('\n...counting done.')

    return words, symbols


def save_counter(counter: Counter, filename: str, *, header='key\tfreq\n') -> None:
    """
    Saves a counter object into a tab delimitered text file.

    """
    with open(filename, 'wt') as dfile:
        dfile.write(header)
        for key, count in counter.most_common():
            dfile.write('{key}\t{count}\n'.format(key=key, count=count))


def load_counter(filename: str) -> Counter:
    """
    Loads a counter out of a tab delimitered text file.

    """
    with open(filename, 'rt') as dfile:
        # skip header
        dfile.readline()
        counter: Counter = Counter()
        for line in dfile:
            key, count = line.strip().split('\t')
            if key in counter.keys():
                raise ValueError("%s contains two instances (words, symbols, ...) of the same spelling." % filename)
            counter[key] = int(count)
    return counter
