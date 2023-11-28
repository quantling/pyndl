"""
pyndl.preprocess
----------------

*pyndl.preprocess* provides functions in order to preprocess data and create
event files from it.

"""
import collections
import gzip
import multiprocessing
import os
import random
import re
import sys
import time
import warnings

from pyndl import io


def bandsample(population, sample_size=50000, *, cutoff=5, seed=None,
               verbose=False):
    """
    Creates a sample of size sample_size out of the population using
    band sampling.

    """
    # make a copy of the population
    # filter all words with freq < cutoff
    population = [(word, freq) for word, freq in population.items() if freq >=
                  cutoff]

    if seed is not None:
        raise NotImplementedError("Reproducable bandsamples by seeding are not properly implemented yet.")

    # shuffle words with same frequency
    rand = random.Random(seed)
    rand.shuffle(population)
    population.sort(key=lambda x: x[1])  # lowest -> highest freq

    step = sum(freq for word, freq in population) / sample_size
    if verbose:
        print("step %.2f" % step)

    accumulator = 0
    index = 0
    sample = list()
    while 0 <= index < len(population):
        word, freq = population[index]
        accumulator += freq
        if verbose:
            print("%s\t%i\t%.2f" % (word, freq, accumulator))
        if accumulator >= step:
            sample.append((word, freq))
            accumulator -= step
            if verbose:
                print("add\t%s\t%.2f" % (word, accumulator))
            del population[index]
            while accumulator >= step and index >= 1:
                index -= 1
                sample.append(population[index])
                accumulator -= step
                if verbose:
                    word, freq = population[index]
                    print("  add\t%s\t%.2f" % (word, accumulator))
                del population[index]
        else:
            # only add to index if no element was removed
            # if element was removed, index points at next element already
            index += 1
            if verbose and index % 1000 == 0:
                print(".", end="")
                sys.stdout.flush()
    sample = collections.Counter({key: value for key, value in sample})
    return sample


def ngrams_to_word(occurrences, n_chars, outfile, remove_duplicates=True):
    """
    Process the occurrences and write them to outfile.

    Parameters
    ----------
    occurrences : sequence of (cues, outcomes) tuples
        cues and outcomes are both strings where underscores and # are
        special symbols.
    n_chars : number of characters (e.g. 2 for bigrams, 3 for trigrams, ...)
    outfile : file handle

    remove_duplicates : bool
        if True make cues and outcomes per event unique

    """
    for cues, outcomes in occurrences:
        if cues and outcomes:
            occurrence = cues + '_' + outcomes
        else:  # take either
            occurrence = cues + outcomes
        phrase_string = "#" + re.sub("_", "#", occurrence) + "#"
        ngrams = (phrase_string[i:(i + n_chars)] for i in
                  range(len(phrase_string) - n_chars + 1))
        if not ngrams or not occurrence:
            continue
        if remove_duplicates:
            ngrams = set(ngrams)
            occurrence = "_".join(set(occurrence.split("_")))
        outfile.write("{}\t{}\n".format("_".join(ngrams), occurrence))


def process_occurrences(occurrences, outfile, *,
                        cue_structure="trigrams_to_word", remove_duplicates=True):
    """
    Process the occurrences and write them to outfile.

    Parameters
    ----------
    occurrences : sequence of (cues, outcomes) tuples
        cues and outcomes are both strings where underscores and # are
        special symbols.
    outfile : file handle

    cue_structure : {'bigrams_to_word', 'trigrams_to_word', 'word_to_word'}

    remove_duplicates : bool
        if True make cues and outcomes per event unique

    """
    if cue_structure == "bigrams_to_word":
        ngrams_to_word(occurrences, 2, outfile, remove_duplicates=remove_duplicates)
    elif cue_structure == "trigrams_to_word":
        ngrams_to_word(occurrences, 3, outfile, remove_duplicates=remove_duplicates)
    elif cue_structure == "word_to_word":
        for cues, outcomes in occurrences:
            if not cues:
                continue
            if remove_duplicates:
                cues = "_".join(set(cues.split("_")))
                outcomes = "_".join(set(outcomes.split("_")))
            outfile.write("{}\t{}\n".format(cues, outcomes))
    else:
        raise NotImplementedError('cue_structure=%s is not implemented yet.' % cue_structure)


def create_event_file(corpus_file,
                      event_file,
                      *,
                      allowed_symbols="all",
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3,),  # number_of_words,
                      cue_structure="trigrams_to_word",
                      lower_case=False,
                      remove_duplicates=True,
                      verbose=False):
    """
    Create an text based event file from a corpus file.

    .. warning::

        '_', '#', and '\t' are removed from the input of the corpus file and
        replaced by a ' ', which is treated as a word boundary.

    Parameters
    ----------
    corpus_file : str
        path where the corpus file is
    event_file : str
        path where the output file will be created
    allowed_symbols : str, function
        all allowed symbols to include in the events as a set of characters.
        The set of characters might be explicit or contains Regex character sets.

        '_', '#', and TAB are special symbols in the event file and will be removed
        automatically. If the corpus file contains these special symbols a warning
        will be given.

        If you want to use all symbols use the special word ``all``.

        These examples define the same allowed symbols::

            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'a-zA-Z'

        or a function indicating which characters to include. The function should
        return `True`, if the passed character is a allowed symbol.

        For example::

            lambda chr: chr in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            lambda chr: ('a' <= chr <= 'z') or ('A' <= chr <= 'Z')

    context_structure : {"document", "paragraph", "line"}

    event_structure : {"line", "consecutive_words", "word_to_word", "sentence"}

    event_options : None or (number_of_words,) or (before, after) or None
        in "consecutive words" the number of words of the sliding window as
        an integer; in "word_to_word" the number of words before and after the
        word of interest each as an integer.
    cue_structure: {"trigrams_to_word", "word_to_word", "bigrams_to_word"}

    lower_case : bool
        should the cues and outcomes be lower cased
    remove_duplicates : bool
        create unique cues and outcomes per event
    verbose : bool

    Notes
    -----
    Breaks / Separators :

        What marks parts, where we do not want to continue learning?

        * ``---end.of.document---`` string?
        * line breaks?
        * empty lines?

        What do we consider one event?

        * three consecutive words?
        * one line of the corpus?
        * everything between two empty lines?
        * everything within one document?

        Should the events be connected to the events before and after?

        No.

    Context :

        A context is a whole document or a paragraph within which we will take
        (three) consecutive words as occurrences or events. The last words of a
        context will not form an occurrence with the first words of the next
        context.

    Occurrence :

        An occurrence or event is will result in one event in the end. This can
        be (three) consecutive words, a sentence, or a line in the corpus file.

    """

    # define functions to remove special chars / symbols
    special_chars = re.compile("[#_\t]")

    def _remove_special_chars_without_warning(line):
        new_line = special_chars.sub(' ', line)
        return new_line

    def _remove_special_chars_with_warning(line):
        nonlocal remove_special_chars
        new_line = special_chars.sub(' ', line)
        if line != new_line:
            warnings.warn('"_", "#", and "\\t" are special symbols and were therefore removed')
            remove_special_chars = _remove_special_chars_without_warning
        return new_line

    remove_special_chars = _remove_special_chars_with_warning

    if callable(allowed_symbols):
        def filter_symbols(line, replace):
            line_copy = list(line)
            for ii in range(len(line)):
                if not allowed_symbols(line[ii]):
                    line_copy[ii] = replace
            return ''.join(line_copy)
    elif allowed_symbols == 'all':
        def filter_symbols(line, replace):
            return line
    else:
        not_in_symbols = re.compile(f"[^{allowed_symbols:s}]")
        def filter_symbols(line, replace):
            return not_in_symbols.sub(replace, line)

    if event_structure not in ('consecutive_words', 'line', 'word_to_word'):
        raise NotImplementedError('This event structure (%s) is not implemented yet.' % event_structure)

    if context_structure not in ('document', 'line'):
        raise NotImplementedError('This context structure (%s) is not implemented yet.' % context_structure)

    if os.path.isfile(event_file):
        raise OSError('%s file exits. Remove file and start again.' % event_file)

    context_pattern = re.compile("(---end.of.document---|---END.OF.DOCUMENT---)")

    if event_structure == 'consecutive_words':
        number_of_words, = event_options
    elif event_structure == 'word_to_word':
        before, after = event_options

    def gen_occurrences(words):
        """
        Make an occurrence out of consecutive words.

        Take all number_of_words number of consecutive words and make an
        occurrence out of it.

        For words = (A, B, C, D); number_of_words = 3 make: (A, ), (A_B, ),
        (A_B_C, ), (B_C_D, ), (C_D, ), (D, )

        """
        if event_structure == 'consecutive_words':
            occurrences = list()
            # can't have more consecutive words than total words
            length = min(number_of_words, len(words))
            # slide window over list of words
            for ii in range(1 - length, len(words)):
                # no consecutive words before first word
                start = max(ii, 0)
                # no consecutive words after last word
                end = min(ii + length, len(words))
                # append (cues, outcomes) with empty outcomes
                occurrences.append(("_".join(words[start:end]), ""))
            return occurrences
        # for words = (A, B, C, D); before = 2, after = 1
        # make: (B, A), (A_C, B), (A_B_D, C), (B_C, D)
        elif event_structure == 'word_to_word':
            occurrences = list()
            for ii, word in enumerate(words):
                # words before the word to a maximum of before
                cues = words[max(0, ii - before):ii]
                # words after the word to a maximum of before
                cues.extend(words[(ii + 1):min(len(words), ii + 1 + after)])
                # append (cues, outcomes)
                occurrences.append(("_".join(cues), word))
            return occurrences
        elif event_structure == 'line':
            #{"trigrams_to_word", "word_to_word", "bigrams_to_word"}
            if cue_structure in ("trigrams_to_word", "bigrams_to_word"):
                # (cues, outcomes) with empty outcomes
                return [('_'.join(words), ''), ]
            else:
                return [('_'.join(words), '_'.join(words)), ]
        else:
            raise ValueError('gen_occurrences should be one of {"consecutive_words", "word_to_word", "line"}')

    def process_line(line):
        """processes one line of text."""
        if lower_case:
            line = line.lower()
        # remove special chars
        line = remove_special_chars(line)
        # replace all weird characters with space
        line = filter_symbols(line, replace=' ')

        return line

    def gen_words(line):
        """generates words out of a line of text."""
        return [word.strip() for word in line.split(" ") if word.strip()]

    def process_words(words):
        """processes one word and makes an occurrence out of it."""
        occurrences = gen_occurrences(words)
        process_occurrences(occurrences, outfile,
                            cue_structure=cue_structure,
                            remove_duplicates=remove_duplicates)

    def process_context(line):
        """called when a context boundary is found."""
        if context_structure == 'document':
            # remove document marker
            line = context_pattern.sub("", line)
        return line

    with open(corpus_file, "rt", encoding="utf-8") as corpus:
        with gzip.open(event_file, "wt", encoding="utf-8") as outfile:
            outfile.write("cues\toutcomes\n")

            words = []
            for ii, line in enumerate(corpus):
                if verbose and ii % 100000 == 0:
                    print(".", end="")
                    sys.stdout.flush()
                    outfile.flush()
                line = line.strip()

                if context_structure == 'line':
                    line = process_line(line)
                    words = gen_words(line)
                    process_words(words)
                else:
                    if context_pattern.search(line) is not None:
                        # process the first context
                        context1, *contexts = context_pattern.split(line)
                        context1 = process_context(context1)

                        if context1.strip():
                            context1 = process_line(context1.strip())
                            words.extend(gen_words(context1))
                        process_words(words)
                        # process in between contexts
                        while len(contexts) > 1:
                            words = []
                            context1, *contexts = contexts
                            context1 = process_context(context1)
                            if context1.strip():
                                context1 = process_line(context1.strip())
                                words.extend(gen_words(context1))
                                process_words(words)
                        # add last part to next context
                        context1 = contexts[0]
                        context1 = process_context(context1)
                        if context1.strip():
                            context1 = process_line(context1.strip())
                            words.extend(gen_words(context1))
                    else:
                        line = process_line(line)
                        words.extend(gen_words(line))

            # write the last context (the rest) when context_structure is not
            # 'line'
            if context_structure != 'line':
                process_words(words)


class JobFilter():
    # pylint: disable=E0202,missing-docstring

    """
    Stores the persistent information over several jobs and exposes a job
    method that only takes the varying parts as one argument.

    .. note::

        Using a closure is not possible as it is not pickable / serializable.

    """

    @staticmethod
    def return_empty_string():
        return ''

    def __init__(self, keep_cues, keep_outcomes, remove_cues, remove_outcomes,
                 cue_map, outcome_map):
        if ((cue_map is not None and remove_cues is not None) or
                (cue_map is not None and keep_cues != 'all') or
                (remove_cues is not None and keep_cues != 'all')):
            raise ValueError('You can either specify keep_cues, remove_cues, or cue_map.')
        if ((outcome_map is not None and remove_outcomes is not None) or
                (outcome_map is not None and keep_outcomes != 'all') or
                (remove_outcomes is not None and keep_outcomes != 'all')):
            raise ValueError('You can either specify keep_outcomes, remove_outcomes, or outcome_map.')

        if cue_map is not None:
            self.cue_map = collections.defaultdict(self.return_empty_string, cue_map)
            self.process_cues = self.process_cues_map
        elif remove_cues is not None:
            self.remove_cues = set(remove_cues)
            self.process_cues = self.process_cues_remove
        elif keep_cues == 'all':
            self.keep_cues = 'all'
            self.process_cues = self.process_cues_all
        else:
            self.keep_cues = keep_cues
            self.process_cues = self.process_cues_keep
        if outcome_map is not None:
            self.outcome_map = collections.defaultdict(self.return_empty_string, outcome_map)
            self.process_outcomes = self.process_outcomes_map
        elif remove_outcomes is not None:
            self.remove_outcomes = set(remove_outcomes)
            self.process_outcomes = self.process_outcomes_remove
        elif keep_outcomes == 'all':
            self.keep_outcomes = 'all'
            self.process_outcomes = self.process_outcomes_all
        else:
            self.keep_outcomes = set(keep_outcomes)
            self.process_outcomes = self.process_outcomes_keep

    def process_cues(self, cues):
        raise NotImplementedError("Needs to be implemented or assigned by a specific method.")

    def process_cues_map(self, cues):
        cues = [self.cue_map[cue] for cue in cues]
        return [cue for cue in cues if cue]

    def process_cues_remove(self, cues):
        return [cue for cue in cues if cue not in self.remove_cues]

    def process_cues_keep(self, cues):
        return [cue for cue in cues if cue in self.keep_cues]

    def process_cues_all(self, cues):
        return cues

    def process_outcomes(self, outcomes):
        raise NotImplementedError("Needs to be implemented or assigned by a specific method.")

    def process_outcomes_map(self, outcomes):
        outcomes = [self.outcome_map[outcome] for outcome in outcomes]
        return [outcome for outcome in outcomes if outcome]

    def process_outcomes_remove(self, outcomes):
        return [outcome for outcome in outcomes if outcome not in self.remove_outcomes]

    def process_outcomes_keep(self, outcomes):
        return [outcome for outcome in outcomes if outcome in self.keep_outcomes]

    def process_outcomes_all(self, outcomes):
        return outcomes

    def job(self, line):
        try:
            cues, outcomes = line.strip('\n').split("\t")
        except ValueError:
            raise ValueError("tabular event file need to have two tab separated columns")
        cues = cues.split("_")
        outcomes = outcomes.split("_")
        cues = self.process_cues(cues)
        outcomes = self.process_outcomes(outcomes)
        # no cues left?
        # NOTE: We want to keep events with no outcomes as this is the
        # background for the cues in that events.
        if not cues:
            return None
        processed_line = ("%s\t%s\n" % ("_".join(cues), "_".join(outcomes)))
        return processed_line


def filter_event_file(input_event_file, output_event_file, *,
                      keep_cues="all", keep_outcomes="all",
                      remove_cues=None, remove_outcomes=None,
                      cue_map=None, outcome_map=None,
                      n_jobs=1, number_of_processes=None, chunksize=100000,
                      verbose=False):
    """
    Filter an event file by a list or a map of cues and outcomes.

    Parameters
    ----------
    You can either use keep_*, remove_*, or map_*.

    input_event_file : str
        path where the input event file is
    output_event_file : str
        path where the output file will be created
    keep_cues : "all" or sequence of str
        list of all cues that should be kept
    keep_outcomes : "all" or sequence of str
        list of all outcomes that should be kept
    remove_cues : None or sequence of str
        list of all cues that should be removed
    remove_outcomes : None or sequence of str
        list of all outcomes that should be removed
    cue_map : dict
        maps every cue as key to the value. Removes all cues that do not have a
        key. This can be used to map several different cues to the same cue or
        to rename cues.
    outcome_map : dict
        maps every outcome as key to the value. Removes all outcome that do not have a
        key. This can be used to map several different outcomes to the same
        outcome or to rename outcomes.
    n_jobs : int
        number of threads to use
    chunksize : int
        number of chunks per submitted job, should be around 100000

    Notes
    -----
    It will keep all cues that are within the event and that (for a human
    reader) might clearly belong to a removed outcome. This is on purpose and
    is the expected behaviour as these cues are in the context of this outcome.

    If an event has no cues it gets removed, but if an event has no outcomes it
    is still present in order to capture the background rate of that cues.

    """
    if number_of_processes is not None:
        warnings.warn("Parameter `number_of_processes` is renamed to `n_jobs`. The old name "
                      "will stop working with v0.9.0.",
                      DeprecationWarning, stacklevel=2)
        n_jobs = number_of_processes
    job = JobFilter(keep_cues, keep_outcomes, remove_cues, remove_outcomes,
                    cue_map, outcome_map)

    with multiprocessing.Pool(n_jobs) as pool:
        with gzip.open(input_event_file, "rt", encoding="utf-8") as infile:
            with gzip.open(output_event_file, "wt", encoding="utf-8") as outfile:
                # copy header
                outfile.write(infile.readline())
                for ii, processed_line, in enumerate(pool.imap(job.job, infile,
                                                               chunksize=chunksize)):
                    if processed_line is not None:
                        outfile.write(processed_line)
                    if verbose and ii % 100000 == 0:
                        print('.', end='')
                        sys.stdout.flush()


################
#  Preprocessing
################

MAGIC_NUMBER = 14159265
CURRENT_VERSION_WITH_FREQ = 215
CURRENT_VERSION = 2048 + 215


def read_binary_file(binary_file_path):
    with open(binary_file_path, "rb") as binary_file:
        magic_number = to_integer(binary_file.read(4))
        if not magic_number == MAGIC_NUMBER:
            raise ValueError('Header does not match the magic number')
        version = to_integer(binary_file.read(4))
        if version == CURRENT_VERSION:
            pass
        else:
            raise ValueError('Version is incorrectly specified')

        nr_of_events = to_integer(binary_file.read(4))
        for _ in range(nr_of_events):
            # Cues
            number_of_cues = to_integer(binary_file.read(4))
            cue_ids = [to_integer(binary_file.read(4)) for ii in range(number_of_cues)]
            # outcomes
            number_of_outcomes = to_integer(binary_file.read(4))
            outcome_ids = [to_integer(binary_file.read(4)) for ii in range(number_of_outcomes)]
            yield (cue_ids, outcome_ids)


def to_bytes(int_):
    return int_.to_bytes(4, 'little')


def to_integer(byte_):
    return int.from_bytes(byte_, "little")


def write_events(events, filename, *, start=0, stop=4294967295, remove_duplicates=None):
    """
    Write out a list of events to a disk file in binary format.

    Parameters
    ----------
    events : iterator of (cue_ids, outcome_ids) tuples called event

    filename : string

    start : first event to write (zero based index)

    stop : last event to write (zero based index; excluded)

    remove_duplicates : {None, True, False}
        if None though a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)

    Returns
    -------
    number_events : int
        actual number of events written to file

    Notes
    -----
    The **binary format** as the following structure::

        8 byte header
        nr of events
        nr of cues in first event
        ids for every cue
        nr of outcomes in first event
        ids for every outcome
        nr of cues in second event
        ...

    Raises
    ------
    StopIteration : events generator is exhausted before stop is reached

    """
    with open(filename, "wb") as out_file:
        # 8 bytes header
        out_file.write(to_bytes(MAGIC_NUMBER))
        out_file.write(to_bytes(CURRENT_VERSION))

        # events
        # estimated number of events (will be rewritten if the actual number
        # differs)
        n_events_estimate = stop - start
        out_file.write(to_bytes(n_events_estimate))

        n_events = 0

        for ii, event in enumerate(events):
            if ii < start:
                continue
            if ii >= stop:
                break

            n_events += 1
            cue_ids, outcome_ids = event

            if remove_duplicates is None:
                if len(cue_ids) != len(set(cue_ids)) or len(outcome_ids) != len(set(outcome_ids)):
                    raise ValueError(''.join([
                        'event %i does not have unique cues or outcomes.'
                        'Use remove_duplicates=True in order to force unique cues and outcomes.'
                        'Use remove_duplicates=False to allow the same cue or outcome multiple'
                        'times in the same event (not recommended)']) % ii)
            elif remove_duplicates:
                cue_ids = set(cue_ids)
                outcome_ids = set(outcome_ids)
            else:
                pass

            # cues in event
            out_file.write(to_bytes(len(cue_ids)))
            for cue_id in cue_ids:
                out_file.write(to_bytes(cue_id))

            # outcomes in event
            out_file.write(to_bytes(len(outcome_ids)))
            for outcome_id in outcome_ids:
                out_file.write(to_bytes(outcome_id))

        if n_events != n_events_estimate and not n_events == 0:
            # the generator was exhausted earlier
            out_file.seek(8)
            out_file.write(to_bytes(n_events))
            raise StopIteration(("event generator was exhausted before stop", n_events))

    if n_events == 0:
        os.remove(filename)
    return n_events


def event_generator(event_file, cue_id_map, outcome_id_map, *, sort_within_event=False):
    for cues, outcomes in io.events_from_file(event_file):
        # uses list and not generators; as generators can only be traversed once
        event = ([cue_id_map[cue] for cue in cues],
                 [outcome_id_map[outcome] for outcome in outcomes])
        if sort_within_event:
            cue_ids, outcome_ids = event
            cue_ids = list(cue_ids)
            cue_ids.sort()
            outcome_ids = list(outcome_ids)
            outcome_ids.sort()
            event = (cue_ids, outcome_ids)
        yield event


def _job_binary_event_file(*,
                           file_name,
                           event_file,
                           cue_id_map,
                           outcome_id_map,
                           sort_within_event,
                           start,
                           stop,
                           remove_duplicates):
    # create generator which is not pickable
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=sort_within_event)
    n_events = write_events(events, file_name, start=start, stop=stop, remove_duplicates=remove_duplicates)
    return n_events


def create_binary_event_files(event_file,
                              path_name,
                              cue_id_map,
                              outcome_id_map,
                              *,
                              sort_within_event=False,
                              n_jobs=2,
                              events_per_file=10000000,
                              overwrite=False,
                              remove_duplicates=None,
                              verbose=False):
    """
    Creates the binary event files for a tabular cue outcome corpus.

    Parameters
    ----------
    event_file : str
        path to tab separated text file that contains all events in a cue
        outcome table.
    path_name : str
        folder name where to store the binary event files
    cue_id_map : dict (str -> int)
        cue to id map
    outcome_id_map : dict (str -> int)
        outcome to id map
    sort_within_event : bool
        should we sort the cues and outcomes within the event
    n_jobs : int
        number of threads to use
    events_per_file : int
        Number of events in each binary file. Has to be larger than 1
    overwrite : bool
        overwrite files if they exist
    remove_duplicates : {None, True, False}
        if None though a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)
    verbose : bool

    Returns
    -------
    number_events : int
        sum of number of events written to binary files
    """
    # pylint: disable=missing-docstring

    if events_per_file < 2:
        raise ValueError("events_per_file has to be larger than 1")

    if not os.path.isdir(path_name):
        if verbose:
            print("create event file folder '%s'" % path_name)
        os.mkdir(path_name, 0o773)
    elif not overwrite:
        raise IOError("folder %s exists and overwrite is False" % path_name)
    else:
        if verbose:
            print("removing event files in '%s'" % path_name)
        for file_name in os.listdir(path_name):
            if "events_0_" in file_name:
                os.remove(os.path.join(path_name, file_name))

    number_events = 0

    with multiprocessing.Pool(n_jobs) as pool:

        def _error_callback(error):
            if isinstance(error, StopIteration):
                _, result = error.value
                nonlocal number_events
                number_events += result  # pylint: disable=undefined-variable
                pool.close()
            else:
                raise error

        def _callback(result):
            nonlocal number_events
            number_events += result
            if verbose:
                print("finished job")
                sys.stdout.flush()

        ii = 0
        while True:
            kwargs = {
                "file_name": os.path.join(path_name, "events_0_%i.dat" % ii),
                "event_file": event_file,
                "cue_id_map": cue_id_map,
                "outcome_id_map": outcome_id_map,
                "sort_within_event": sort_within_event,
                "start": ii*events_per_file,
                "stop": (ii+1)*events_per_file,
                "remove_duplicates": remove_duplicates,
            }
            try:
                result = pool.apply_async(_job_binary_event_file,
                                          kwds=kwargs,
                                          callback=_callback,
                                          error_callback=_error_callback)
                if verbose:
                    print("submitted job %i" % ii)
            except ValueError as error:
                # someone has closed the pool with the correct error callback
                if error.args[0] == 'Pool not running':
                    if verbose:
                        print("reached end of events")
                    break  # out of while True
                else:
                    raise error
            ii += 1
            # only start jobs in chunks of 4*n_jobs
            if ii % (n_jobs*4) == 0:
                while True:
                    if result.ready():
                        break
                    time.sleep(1.0)  # check every second
                    if verbose:
                        print('c')
                        sys.stdout.flush()
        # wait until all jobs are done
        pool.close()
        pool.join()
        if verbose:
            print("finished all jobs.\n")
    return number_events

# for example code see function test_preprocess in file
# ./tests/test_preprocess.py.
