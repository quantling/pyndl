# !/usr/bin/env/python3
# coding: utf-8

from collections import Counter
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
    if seed is not None:
        random.seed(seed)
    random.shuffle(population)
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
            if index % 10000 == 0:
                print(".", end="")
                sys.stdout.flush()
    sample = Counter({key: value for key, value in sample})
    return sample


def process_occurrences(occurrences, outfile):
    """
    occurrences : sequence of str
    outfile : file handle

    """
    for occurence in occurrences:
        if occurence:
            phrase_string = "#" + re.sub("_", "#", occurence) + "#"
            trigrams = (phrase_string[i:i+3] for i in
                        range(len(phrase_string)-2))
            outfile.write("_".join(trigrams) + "\t" + occurence + "\t1\n")


def create_event_file(corpus_file,
                      event_file,
                      symbols="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      *,
                      context="document",
                      event="consecutive_words",
                      event_option=3,
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
    context : {"document", "paragraph"}
    event : {"line", "consecutive_words", "sentence"}


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
    if "_" in symbols or "#" in symbols:
        raise ValueError("_ and # are special symbols and cannot be in symbols string")

    in_symbols = re.compile("^[%s]*$" % symbols)
    not_in_symbols = re.compile("[^%s]" % symbols)
    document_pattern = re.compile("---end.of.document---")

    with open(corpus_file, "rt") as corpus:
        with open(event_file, "wt") as outfile:
            outfile.write("cues\toutcomes\tfrequency\n")

            occurrences = []
            contexts = []

            if context == "document" and event == "consecutive_words":
                words = []

                for ii, line in enumerate(corpus):
                    if verbose and ii % 100000 == 0:
                        print(".", end="")
                        sys.stdout.flush()
                    if lower_case:
                        line = line.lower().strip()
                    else:
                        line = line.strip()

                    if document_pattern.match(line.lower()) is not None:

                        if len(words) < event_option:
                            occurrences.append("_".join(words))
                        else:
                            # take all event_option number of consecutive words and make an occurence out of it.
                            for jj in range(len(words) - (event_option - 1)):
                                occurrences.append("_".join(words[jj:(jj+event_option)]))
                        process_occurrences(occurrences, outfile)
                        occurrences = []
                        words = []

                    # replace all weird characters with space
                    line = not_in_symbols.sub(" ", line)

                    words.extend([word.strip() for word in line.split(" ") if word.strip()])
            else:
                raise NotImplementedError("This combination of context=%s and event=%s is not implemented yet." % (str(context), str(event)))


def filter_event_file(input_event_file, output_event_file, allowed_cues="all",
                      allowed_outcomes="all"):
    """
    Filter an event file by allowed cues and outcomes.

    Parameters
    ==========
    input_event_file : str
        path where the input event file is
    output_event_file : str
        path where the output file will be created
    allowed_cues : "all" or sequence of str
        list all allowed cues
    allowed_outcomes : "all" or sequence of str
        list all allowed outcomes

    Notes
    =====
    It will keep all cues that are within the event and that (for a human
    reader) might clearly belong to a removed outcome. This is on purpose and
    is the expected behaviour as these cues are in the context of this outcome.

    """
    with open(input_event_file, "rt") as infile:
        with open(output_event_file, "wt") as outfile:
            # copy header
            outfile.write(infile.readline())

            for line in infile:
                try:
                    cues, outcomes, frequency = line.strip().split("\t")
                except ValueError:
                    raise ValueError("tabular event file need to have three tab separated columns")
                cues = cues.split("_")
                outcomes = outcomes.split("_")
                frequency = int(frequency)
                if not allowed_cues == "all":
                    cues = [cue for cue in cues if cue in allowed_cues]
                if not allowed_outcomes == "all":
                    outcomes = [outcome for outcome in outcomes if outcome in allowed_outcomes]
                # no cues or no outcomes left?
                if not cues or not outcomes:
                    continue
                outfile.write("%s\t%s\t%i\n" % ("_".join(cues), "_".join(outcomes), frequency))



################
## Preprocessing
################

MAGIC_NUMBER = 14159265
CURRENT_VERSION = 215


def to_bytes(int_):
    return int_.to_bytes(4, 'little')


def write_events(filename, events, *, start=0, stop=4294967296):
    """
    Write out a list of events to a disk file in binary format.

    This resembles the function ``writeEvents`` in
    ndl2/src/common/serialization.cpp.

    Parameters
    ==========
    filename : string
    events : iterator of (cue_ids, outcome_ids, frequency) triples called event
    start : first event to write (zero based index)
    stop : last event to write (zero based index; excluded)

    Binary Format
    =============

    ::

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
            out_file.write(to_bytes(frequency))

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
                cues, outcomes, frequency = line.strip().split("\t")
            except ValueError:
                raise ValueError("tabular corpus file need to have three tab separated columns")
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


def _job_binary_event_files(*, filename, event_file, cue_id_map,
                            outcome_id_map, sort_within_event, start, stop):
    # create generator which is not pickable
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=sort_within_event)
    write_events(filename, events, start=start, stop=stop)


def create_binary_event_files(path_name, event_file, cue_id_map,
                              outcome_id_map,
                              *, sort_within_event=False, number_of_processes=2,
                              events_per_file=1000000, overwrite=False,
                              verbose=False):
    """
    Creates the binary event files for a tabular cue outcome frequency corpus.

    Parameters
    ==========
    path_name : str
        folder name where to store the binary event files
    event_file : str
        path to tab separated text file that contains all events in a cue
        outcome frequency table.
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
            kwargs = {"filename": os.path.join(path_name, "events_0_%i.dat" % ii),
                      "event_file": event_file,
                      "cue_id_map": cue_id_map,
                      "outcome_id_map": outcome_id_map,
                      "sort_within_event": sort_within_event,
                      "start": ii*events_per_file,
                      "stop": (ii+1)*events_per_file}
            try:
                #result = pool.apply(_job_binary_event_files, kwds=kwargs)
                result = pool.apply_async(_job_binary_event_files,
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
                    time.sleep(1)
        # wait until all jobs are done
        pool.close()
        pool.join()
        print("finished all jobs.\n")


if __name__ == "__main__":

    from .count import cues_outcomes

    corpus_file = "./tests/corpus.txt"
    event_file = "./tests/events_corpus.tab"
    symbols = "abcdefghijklmnopqrstuvwxyzóąćęłńśźż"  # polish

    #symbols = "aâăbcdefghiîjklmnopqrsştţuvwxyz")  # romanian

    create_eventfile(corpus_file, event_file, symbols,
                     context="document", event="consecutive_words",
                     event_option=3, lower_case=True, verbose=True)

    cue_freq_map, outcome_freq_map = cues_outcomes(event_file,
                                                   number_of_processes=2)
    cues = list(cue_freq_map.keys())
    cues.sort()
    cue_id_map = {cue: ii for ii, cue in enumerate(cues)}

    outcome_freq_map_filtered = bandsample(outcome_freq_map, 50, cutoff=1)
    outcomes = list(outcome_freq_map_filtered.keys())
    outcomes.sort()
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}

    event_file_filtered = event_file + ".filtered"
    filter_eventfile(event_file, event_file_filtered, allowed_outcomes=outcomes)


    path_name = event_file_filtered + ".events"
    create_binary_event_files(path_name, event_file_filtered, cue_id_map,
                              outcome_id_map, sort_within_event=False,
                              number_of_processes=2, events_per_file=100, overwrite=True,
                              verbose=True)


