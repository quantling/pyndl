"""
pyndl.io
--------

*pyndl.io* provides functions to create event generators from different
sources in order to use them with *pyndl.ndl* to train NDL models or to save
existing events from a DataFrame or a list to a file.

"""

import gzip
from collections import Iterator, Iterable

import pandas as pd


def events_from_file(event_path, compression="gzip"):
    """
    Yields events for all events in a gzipped event file.

    Parameters
    ----------
    event_path : str
        path to gzipped event file
    compression : str
        indicates whether the events should be read from gunzip
        file or not can be {"gzip" or None}

    Yields
    ------
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    if compression == "gzip":
        event_file = gzip.open(event_path, 'rt')
    elif compression is None:
        event_file = open(event_path, 'rt')
    else:
        raise ValueError("compression needs to be 'gzip' or None")

    try:
        # skip header
        event_file.readline()
        for line in event_file:
            cues, outcomes = line.strip('\n').split('\t')
            cues = cues.split('_')
            outcomes = outcomes.split('_')
            yield (cues, outcomes)
    finally:
        event_file.close()


def events_to_file(events, file_path, delimiter="\t", compression="gzip",
                   columns=("cues", "outcomes")):
    """
    Writes events to a file

    Parameters
    ----------
    events : pandas.DataFrame or Iterator or Iterable
        a pandas DataFrame with one event per row and one colum with the cues
        and one column with the outcomes or a list of cues and outcomes as strings
        or a list of a list of cues and a list of outcomes which should be written
        to a file
    file_path: str
        path to where the file should be saved
    delimiter: str
        Seperator which should be used. Default ist a tab
    compression : str
        indicates whether the events should be read from gunzip
        file or not can be {"gzip" or None}
    columns: tuple
        a tuple of column names

    """
    if isinstance(events, pd.DataFrame):
        events = events_from_dataframe(events)
    elif isinstance(events, (Iterator, Iterable)):
        events = events_from_list(events)
    else:
        raise ValueError("events should either be a pd.DataFrame or an Iterator or an Iterable.")

    if compression == "gzip":
        out_file = gzip.open(file_path, 'wt')
    elif compression is None:
        out_file = open(file_path, 'wt')
    else:
        raise ValueError("compression needs to be 'gzip' or None")

    try:
        out_file.write("{}\n".format(delimiter.join(columns)))

        for cues, outcomes in events:
            if isinstance(cues, list) and isinstance(outcomes, list):
                line = "{}{}{}\n".format("_".join(cues),
                                         delimiter,
                                         "_".join(outcomes))
            elif isinstance(cues, str) and isinstance(outcomes, str):
                line = "{}{}{}\n".format(cues, delimiter, outcomes)
            else:
                raise ValueError("cues and outcomes should either be a list or a string.")
            out_file.write(line)
    finally:
        out_file.close()


def events_from_dataframe(df, columns=("cues", "outcomes")):
    """
    Yields events for all events in a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        a pandas DataFrame with one event per row and one colum with the cues
        and one column with the outcomes.
    columns : tuple
        a tuple of column names

    Yields
    ------
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    for _, row in df.iterrows():
        cues, outcomes = row[list(columns)]
        cues = cues.split('_')
        outcomes = outcomes.split('_')
        yield (cues, outcomes)


def events_from_list(lst):
    """
    Yields events for all events in a list.

    Parameters
    ----------
    lst : list of list of str or list of str
        a list either containing a list of cues as strings and a list of outcomes
        as strings or a list containing a cue and an outcome string, where cues
        respectively outcomes are seperated by an undescore

    Yields
    ------
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    for cues, outcomes in lst:
        if isinstance(cues, str):
            cues = cues.split('_')
        if isinstance(outcomes, str):
            outcomes = outcomes.split('_')
        yield (cues, outcomes)
