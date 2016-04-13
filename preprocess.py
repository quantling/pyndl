# !/usr/bin/env/python3
# coding: utf-8

import collections
import multiprocessing
import os
import random
import re
import sys
import time


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
    # shuffle words with same frequency
    rand = random.Random(seed)  # TODO not working properly :(
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


def process_occurrences(occurrences, outfile, *,
        cue_structure="trigrams_to_word"):
    """
    Process the occurrences and write them to outfile.

    Parameters
    ==========
    occurrences : sequence of (cues, outcomes) tuples
        cues and outcomes are both strings where underscores and # are
        special symbols.
    outfile : file handle
    cue_structure : {'bigrams_to_word', 'trigrams_to_word', 'word_to_word'}

    """
    if cue_structure == "bigrams_to_word":
        for cues, outcomes in occurrences:
            if cues and outcomes:
                occurrence = cues + '_' + outcomes
            else:  # take either
                occurrence = cues + outcomes
            phrase_string = "#" + re.sub("_", "#", occurrence) + "#"
            bigrams = (phrase_string[i:(i + 2)] for i in
                        range(len(phrase_string) - 2 + 1))
            if not bigrams or not occurrence:
                continue
            outfile.write("_".join(bigrams) + "\t" + occurrence + "\t1\n")
    elif cue_structure == "trigrams_to_word":
        for cues, outcomes in occurrences:
            if cues and outcomes:
                occurrence = cues + '_' + outcomes
            else:  # take either
                occurrence = cues + outcomes
            phrase_string = "#" + re.sub("_", "#", occurrence) + "#"
            trigrams = (phrase_string[i:(i + 3)] for i in
                        range(len(phrase_string) - 3 + 1))
            if not trigrams or not occurrence:
                continue
            outfile.write("_".join(trigrams) + "\t" + occurrence + "\t1\n")
    elif cue_structure == "word_to_word":
        for cues, outcomes in occurrences:
            if not cues:
                continue
            outfile.write(cues + "\t" + outcomes + "\t1\n")
    else:
        raise NotImplementedError('cue_structure=%s is not implemented yet.' % cue_structure)


def create_event_file(corpus_file,
                      event_file,
                      symbols="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      *,
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3,),  # number_of_words,
                      cue_structure="trigrams_to_word",
                      lower_case=False,
                      verbose=False):
    """
    Create an text based event file from a corpus file.

    Parameters
    ==========
    corpus_file : str
        path where the corpus file is
    event_file : str
        path where the output file will be created
    symbols : str
        string of all valid symbols
    context_structure : {"document", "paragraph", "line"}
    event_structure : {"line", "consecutive_words", "word_to_word", "sentence"}
    event_options : None or (number_of_words,) or (before, after) or None
        in "consecutive words" the number of words of the sliding window as
        an integer; in "word_to_word" the number of words before and after the
        word of interst each as an integer.
    cue_structure: {"trigrams_to_word", "word_to_word", "bigrams_to_word"}
    lower_case : bool
        should the cues and outcomes be lower cased
    verbose : bool

    Breaks / Separators
    ===================
    What marks parts, where we do not want to continue learning?

    * ---end.of.document--- string?
    * line breaks?
    * empty lines?

    What do we consider one event?

    * three consecutive words?
    * one line of the corpus?
    * everything between two empty lines?
    * everything within one document?

    Should the events be connected to the events before and after?

    No.

    Context
    =======
    A context is a whole document or a paragraph within which we will take
    (three) consecutive words as occurrences or events. The last words of a
    context will not form an occurrence with the first words of the next
    context.

    Occurrence
    ==========
    An occurrence or event is will result in one event in the end. This can be
    (three) consecutive words, a sentence, or a line in the corpus file.

    """
    if '_' in symbols or '#' in symbols or '\t' in symbols:
        raise ValueError('"_", "#", and "\\t" are special symbols and cannot be in symbols string')

    if event_structure not in ('consecutive_words', 'line', 'word_to_word'):
        raise NotImplementedError('This event structure (%s) is not implemented yet.' % event_structure)

    if context_structure not in ('document', 'line'):
        raise NotImplementedError('This context structure (%s) is not implemented yet.' % context_structure)

    if os.path.isfile(event_file):
        raise OSError('%s file exits. Remove file and start again.' % event_file)

    in_symbols = re.compile("^[%s]*$" % symbols)
    not_in_symbols = re.compile("[^%s]" % symbols)
    context_pattern = re.compile("(---end.of.document---|---END.OF.DOCUMENT---)")

    if event_structure == 'consecutive_words':
        number_of_words, = event_options
    elif event_structure == 'word_to_word':
        before, after = event_options

    def gen_occurrences(words):
        # take all number_of_words number of consecutive words and make an
        # occurrence out of it.
        # for words = (A, B, C, D); number_of_words = 3
        # make: (A, ), (A_B, ), (A_B_C, ), (B_C_D, ), (C_D, ), (D, )
        if event_structure == 'consecutive_words':
            occurrences = list()
            cur_words = list()
            ii = 0
            while True:
                if ii < len(words):
                    cur_words.append(words[ii])
                if ii >= len(words) or ii >= number_of_words:
                    # remove the first word
                    cur_words = cur_words[1:]
                # append (cues, outcomes) with empty outcomes
                occurrences.append(("_".join(cur_words), ''))
                ii += 1
                if not cur_words:
                    break
            return occurrences
        # for words = (A, B, C, D); before = 2, after = 1
        # make: (B, A), (A_C, B), (A_B_D, C), (B_C, D)
        elif event_structure == 'word_to_word':
            occurrences = list()
            for ii, word in enumerate(words):
                # words before the word to a maximum of before
                cues = words[max(0, ii - before):ii]
                # words after the word to a maximum of before
                cues.extend(
                        words[(ii + 1):min(len(words), ii + 1 + after)])
                # append (cues, outcomes)
                occurrences.append(("_".join(cues), word))
            return occurrences
        elif event_structure == 'line':
            # (cues, outcomes) with empty outcomes
            return [('_'.join(words), ''),]

    def process_line(line):
        if lower_case:
            line = line.lower()
        # replace all weird characters with space
        line = not_in_symbols.sub(" ", line)
        return line

    def gen_words(line):
        return [word.strip() for word in line.split(" ") if word.strip()]

    def process_words(words):
        occurrences = gen_occurrences(words)
        process_occurrences(occurrences, outfile,
                            cue_structure=cue_structure)

    def process_context(line):
        '''called when a context boundary is found.'''
        if context_structure == 'document':
            # remove document marker
            line = context_pattern.sub("", line)
        return line

    with open(corpus_file, "rt") as corpus:
        with open(event_file, "wt") as outfile:
            outfile.write("cues\toutcomes\tfrequency\n")

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

            # write the last context (the rest)!
            if not context_structure == 'line':
                process_words(words)



class JobFilter():
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
        if ((cue_map is not None and remove_cues is not None)
                or (cue_map is not None and keep_cues != 'all')
                or (remove_cues is not None and keep_cues != 'all')):
            raise ValueError('You can either specify keep_cues, remove_cues, or cue_map.')
        if ((outcome_map is not None and remove_outcomes is not None)
                or (outcome_map is not None and keep_outcomes != 'all')
                or (remove_outcomes is not None and keep_outcomes != 'all')):
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
            cues, outcomes, frequency = line.split("\t")
        except ValueError:
            raise ValueError("tabular event file need to have three tab separated columns")
        cues = cues.split("_")
        outcomes = outcomes.split("_")
        frequency = int(frequency)
        cues = self.process_cues(cues)
        outcomes = self.process_outcomes(outcomes)
        # no cues left?
        # NOTE: We want to keep events with no outcomes as this is the
        # background for the cues in that events.
        if not cues:
            return None
        processed_line = ("%s\t%s\t%i\n" % ("_".join(cues), "_".join(outcomes), frequency))
        return processed_line


def filter_event_file(input_event_file, output_event_file, *,
                      keep_cues="all", keep_outcomes="all",
                      remove_cues=None, remove_outcomes=None,
                      cue_map=None, outcome_map=None,
                      number_of_processes=1, chunksize=100000,
                      verbose=False):
    """
    Filter an event file by a list or a map of cues and outcomes.

    Parameters
    ==========
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
    number_of_processes : int
        number of threads to use
    chunksize : int
        number of chunks per submitted job, should be around 100000

    Notes
    =====
    It will keep all cues that are within the event and that (for a human
    reader) might clearly belong to a removed outcome. This is on purpose and
    is the expected behaviour as these cues are in the context of this outcome.

    If an event has no cues it gets removed, but if an event has no outcomes it
    is still present in order to capture the background rate of that cues.

    """
    job = JobFilter(keep_cues, keep_outcomes, remove_cues, remove_outcomes,
                    cue_map, outcome_map)

    with multiprocessing.Pool(number_of_processes) as pool:
        with open(input_event_file, "rt") as infile:
            with open(output_event_file, "wt") as outfile:
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
## Preprocessing
################

MAGIC_NUMBER = 14159265
CURRENT_VERSION_WITH_FREQ = 215
CURRENT_VERSION = 2048 + 215


def to_bytes(int_):
    return int_.to_bytes(4, 'little')


def write_events(events, filename, *, start=0, stop=4294967295, store_freq=False):
    """
    Write out a list of events to a disk file in binary format.

    This resembles the function ``writeEvents`` in
    ndl2/src/common/serialization.cpp.

    Parameters
    ==========
    events : iterator of (cue_ids, outcome_ids, frequency) triples called event
    filename : string
    start : first event to write (zero based index)
    stop : last event to write (zero based index; excluded)
    store_freq : bool
        should the frequency information per event be stored, if False all
        frequencies need to be 1.

    Binary Format
    =============
    Without frequency information::

        8 byte header
        nr of events
        nr of cues in first event
        ids for every cue
        nr of outcomes in first event
        ids for every outcome
        frequency
        nr of cues in second event
        ...


    With frequency information (old format)::

        8 byte header
        nr of events
        nr of cues in first event
        ids for every cue
        nr of outcomes in first event
        ids for every outcome
        frequency
        nr of cues in second event
        ...

    Raises
    ======
    StopIteration : events generator is exhausted before stop is reached

    """

    with open(filename, "wb") as out_file:
        # 8 bytes header
        out_file.write(to_bytes(MAGIC_NUMBER))
        if store_freq:
            out_file.write(to_bytes(CURRENT_VERSION_WITH_FREQ))
        else:
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
            cue_ids, outcome_ids, frequency = event

            # cues in event
            out_file.write(to_bytes(len(cue_ids)))
            for cue_id in cue_ids:
                out_file.write(to_bytes(cue_id))

            # outcomes in event
            out_file.write(to_bytes(len(outcome_ids)))
            for outcome_id in outcome_ids:
                out_file.write(to_bytes(outcome_id))

            # frequency
            if store_freq:
                out_file.write(to_bytes(frequency))
            else:
                if frequency != 1:
                    raise ValueError('All frequencies need to 1 or use store_freq=True.')

        if n_events != n_events_estimate and not n_events == 0:
            # the generator was exhausted earlier
            out_file.seek(8)
            out_file.write(to_bytes(n_events))
            raise StopIteration("event generator was exhausted before stop")

    if n_events == 0:
        os.remove(filename)



def event_generator(event_file, cue_id_map, outcome_id_map, *, sort_within_event=False):
    with open(event_file, "rt") as in_file:
        # skip header
        in_file.readline()
        for nn, line in enumerate(in_file):
            try:
                cues, outcomes, frequency = line.split("\t")
            except ValueError:
                raise ValueError("tabular event file need to have three tab separated columns")
            cues = cues.split("_")
            outcomes = outcomes.split("_")
            frequency = int(frequency)
            # uses list and not generators; as generators can only be traversed once
            event = ([cue_id_map[cue] for cue in cues],
                     [outcome_id_map[outcome] for outcome in outcomes],
                     frequency)
            if sort_within_event:
                cue_ids, outcome_ids, frequency = event
                cue_ids = list(cue_ids)
                cue_ids.sort()
                outcome_ids = list(outcome_ids)
                outcome_ids.sort()
                event = (cue_ids, outcome_ids, frequency)
            yield event


def _job_binary_event_file(*, file_name, event_file, cue_id_map,
                            outcome_id_map, sort_within_event, start, stop, store_freq):
    # create generator which is not pickable
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=sort_within_event)
    write_events(events, file_name, start=start, stop=stop, store_freq=store_freq)


def create_binary_event_files(event_file, path_name, cue_id_map,
                              outcome_id_map,
                              *, sort_within_event=False, number_of_processes=2,
                              events_per_file=1000000, overwrite=False, store_freq=False,
                              verbose=False):
    """
    Creates the binary event files for a tabular cue outcome frequency corpus.

    Parameters
    ==========
    event_file : str
        path to tab separated text file that contains all events in a cue
        outcome frequency table.
    path_name : str
        folder name where to store the binary event files
    cue_id_map : dict (str -> int)
        cue to id map
    outcome_id_map : dict (str -> int)
        outcome to id map
    sort_within_event : bool
        should we sort the cues and outcomes within the event
    number_of_processes : int
        number of threads to use
    events_per_file : int
    overwrite : overwrite files if they exist
    store_freq : bool
        should the frequency be stored, otherwise it needs to be 1 for all events
    verbose : bool

    """

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

    with multiprocessing.Pool(number_of_processes) as pool:

        def error_callback(error):
            if isinstance(error, StopIteration):
                pool.close()
            else:
                raise error

        def callback(result):
            if verbose:
                print("finished job")
                sys.stdout.flush()

        ii = 0
        while True:
            kwargs = {"file_name": os.path.join(path_name, "events_0_%i.dat" % ii),
                      "event_file": event_file,
                      "cue_id_map": cue_id_map,
                      "outcome_id_map": outcome_id_map,
                      "sort_within_event": sort_within_event,
                      "start": ii*events_per_file,
                      "stop": (ii+1)*events_per_file,
                      "store_freq": store_freq,}
            try:
                result = pool.apply_async(_job_binary_event_file,
                                          kwds=kwargs,
                                          callback=callback,
                                          error_callback=error_callback)
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
            # only start jobs in chunks of 4*number_of_processes
            if ii % (number_of_processes*4) == 0:
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
        print("finished all jobs.\n")


# for example code see function test_preprocess in file
# ./tests/test_preprocess.py.

