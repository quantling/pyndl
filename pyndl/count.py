"""
pyndl.count
-----------

*pyndl.count* provides functions in order to count

* words and symbols in a corpus file
* cues and outcomes in an event file

"""
# pylint: disable=redefined-outer-name, invalid-name

from collections import Counter, namedtuple
import gzip
import itertools
import multiprocessing
import sys
import warnings

from pyndl import io


CuesOutcomes = namedtuple('CuesOutcomes', 'n_events, cues, outcomes')
WordsSymbols = namedtuple('WordsSymbols', 'words, symbols')


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
    events = io.events_from_file(event_file_name, start=start, step=step)
    for nn, (cue_list, outcome_list) in enumerate(events):
        for cue in cue_list:
            cues[cue] += 1
        for outcome in outcome_list:
            outcomes[outcome] += 1
        if verbose and nn % 100000 == 0:
            print('.', end='')
            sys.stdout.flush()
    return (nn + 1, cues, outcomes)


def cues_outcomes(event_file_name,
                  *, n_jobs=2, number_of_processes=None, verbose=False):
    """
    Counts cues and outcomes in event_file_name using n_jobs
    processes.

    Returns
    -------
    (n_events, cues, outcomes) : (int, collections.Counter, collections.Counter)

    """
    if number_of_processes is not None:
        warnings.warn("Parameter `number_of_processes` is renamed to `n_jobs`. The old name "
                      "will stop working with v0.9.0.",
                      DeprecationWarning, stacklevel=2)
        n_jobs = number_of_processes
    with multiprocessing.Pool(n_jobs) as pool:
        step = n_jobs
        results = pool.starmap(_job_cues_outcomes,
                               ((event_file_name,
                                 start,
                                 step,
                                 verbose)
                                for start in range(n_jobs)))
        n_events = 0
        cues = Counter()
        outcomes = Counter()
        for nn, cues_process, outcomes_process in results:
            n_events += nn
            cues += cues_process
            outcomes += outcomes_process

    if verbose:
        print('\n...counting done.')

    return CuesOutcomes(n_events, cues, outcomes)


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
    with open(corpus_file_name, 'rt', encoding="utf-8") as dfile:
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


def words_symbols(corpus_file_name,
                  *, n_jobs=2, number_of_processes=None, lower_case=False, verbose=False):
    """
    Counts words and symbols in corpus_file_name using n_jobs
    processes.

    Returns
    -------
    (words, symbols) : (collections.Counter, collections.Counter)

    """
    if number_of_processes is not None:
        warnings.warn("Parameter `number_of_processes` is renamed to `n_jobs`. The old name "
                      "will stop working with v0.9.0.",
                      DeprecationWarning, stacklevel=2)
        n_jobs = number_of_processes
    with multiprocessing.Pool(n_jobs) as pool:
        step = n_jobs
        results = pool.starmap(_job_words_symbols, ((corpus_file_name,
                                                     start,
                                                     step,
                                                     lower_case,
                                                     verbose)
                                                    for start in
                                                    range(n_jobs)))
        words = Counter()
        symbols = Counter()
        for words_process, symbols_process in results:
            words += words_process
            symbols += symbols_process

    if verbose:
        print('\n...counting done.')

    return WordsSymbols(words, symbols)


def save_counter(counter, filename, *, header='key\tfreq\n'):
    """
    Saves a counter object into a tab delimitered text file.

    """
    with open(filename, 'wt', encoding="utf-8") as dfile:
        dfile.write(header)
        for key, count in counter.most_common():
            dfile.write('{key}\t{count}\n'.format(key=key, count=count))


def load_counter(filename):
    """
    Loads a counter out of a tab delimitered text file.

    """
    with open(filename, 'rt', encoding="utf-8") as dfile:
        # skip header
        dfile.readline()
        counter = Counter()
        for line in dfile:
            key, count = line.strip().split('\t')
            if key in counter.keys():
                raise ValueError("%s contains two instances (words, symbols, ...) of the same spelling." % filename)
            counter[key] = int(count)
    return counter
