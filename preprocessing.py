# !/usr/bin/env/python3
# coding: utf-8

import re

# corpus = open("/home/christian/test", "r")
# corpus = open("/home/tino/collab-petar/pl.subtitles.utf8", "r")
corpus = open("/home/tino/collab-petar/ro.subtitles.utf8", "r")
# eventfile = open("/home/christian/testevents", "w")
# eventfile = open("/home/tino/debug-ndl2/polish/pl.subtitles.events.utf8", "w", encoding = "utf-8")
eventfile = open("/home/tino/debug-ndl2/polish/ro.subtitles.events.utf8", "w", encoding = "utf-8")

document = []
# regular = re.compile("^[abcdefghijklmnopqrstuvwxyzóąćęłńśźż]*$")     # polish
regular = re.compile(u"^[aâăbcdefghiîjklmnopqrsştţuvwxyz]*$")        # romanian

# write header
eventfile.write("cues\toutcomes\tfrequency\n")

for line in corpus:
    current = line.lower().strip()
    if regular.search(current) is None:
        # make events out of each document
        if "end.of.document" in current:
            # make events and write to file
            for i in range(len(document)-2):
                phrase = document[i:i+3]
                phraseString = "#" + "#".join(phrase) + "#"
                trigrams = [phraseString[i:i+3] for i in range(len(phraseString)-2)]
                eventfile.write("_".join(trigrams) + "\t" + "_".join(phrase) + "\t1\n")
            document = []
    else:
        document.append(current)

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

attach_one("ro.subtitles.events.utf8")


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
    with open(event_file, "wb") as out_file:





def main():
    corpus_file = ""
    cue_freq_map, outcome_freq_map = extract_cues_outcomes(corpus_file)

    outcomes = read_outcomes(outcome_file)
    outcomes.sort()
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}


