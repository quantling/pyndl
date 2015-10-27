# !/usr/bin/env/python3
# coding: utf-8

import re


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


def create_eventfile(corpus_file,
                     event_file,
                     symbols="abcdefghijklmnopqrstuvwxyz",
                     *,
                     context="document",
                     event="consecutive_words",
                     event_option=3,
                     lower_case=True,
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



import sys
def attach_one(filename):
    with open(filename + ".withone", "wt") as out_file:
        out_file.write("cues\toutcomes\tfrequency\n")
        with open(filename, "rt") as in_file:
            for ii, line in enumerate(in_file):
                out_file.write(line.strip() + "\t1\n")
                if ii % 100000 == 0:
                    print(".", end="")
                    sys.stdout.flush()

if __name__ == "__main__":
    #attach_one("ro.subtitles.events.utf8")

    #corpus_file = "/home/tino/collab-petar/pl.subtitles.utf8"
    #event_file = "/home/tino/collab-petar/pl.subtitles.events.utf8"
    corpus_file = "./tests/corpus.txt"
    event_file = "./tests/events_corpus.tab"
    symbols = "abcdefghijklmnopqrstuvwxyzóąćęłńśźż"  # polish
    #symbols = "aâăbcdefghiîjklmnopqrsştţuvwxyz")  # romanian

    create_eventfile(corpus_file, event_file, symbols,
                     context="document", event="consecutive_words",
                     event_option=3, lower_case=True, verbose=True)


################
## Preprocessing
################

MAGIC_NUMBER = 14159265
CURRENT_VERSION = 215


def to_bytes(int_):
    return int_.to_bytes(4, 'little')


def write_events(filename, events):
    """
    Write out a list of events to a disk file in binary format.

    This resembles the function ``writeEvents`` in
    ndl2/src/common/serialization.cpp.

    Parameters
    ----------

    filename : string

    events : iterator of (cue_ids, outcome_ids, frequency) triples called event

    Binary Format
    -------------

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

    """

    with open(filename, "wb") as out_file:
        # 8 bytes header
        out_file.write(to_bytes(MAGIC_NUMBER))
        out_file.write(to_bytes(CURRENT_VERSION))

        # events
        out_file.write(to_bytes(len(events)))
        for event in events:
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

def event_generator(corpus_file, cue_id_map, outcome_id_map, *, sort_within_event=False):
    with open(corpus_file, "rt") as in_file:
        # skip header
        in_file.readline()
        for nn, line in enumerate(in_file):
            try:
                cues, outcomes, frequency = line.strip().split("\t")
            except ValueError:
                raise ValueError("tabular corpus file need to have three tab separated columns")
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




def process_tabular_corpus(corpus_file, cue_id_map, outcome_id_map, path_name,
                           *, sort_within_event=False, n_threads=2):
    """
    Creates the binary event file format for a tabular cue outcome frequency corpus.

    Parameters
    ----------
    corpus_file : string
        path to tab separated text file that contains all events in a cue
        outcome frequency table.
    cue_id_map : dict (str -> int)
        cue to id map
    outcome_id_map : dict (str -> int)
        outcome to id map
    path_name : str
        folder name where to store the binary event files
    sort_within_event : bool
        should we sort the cues and outcomes within the event
    n_threads : int
        number of threads to use

    """
    event_file = os.path.join(path_name, "events_0_1.dat")

    raise NotImplementedError()
    #with open(event_file, "wb") as out_file:





def main():
    corpus_file = ""
    cue_freq_map, outcome_freq_map = extract_cues_outcomes(corpus_file)

    outcomes = read_outcomes(outcome_file)
    outcomes.sort()
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}


