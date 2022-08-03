"""
pyndl.corpus
------------

*pyndl.corpus* generates a corpus file (outfile) out of a bunch of gunzipped xml
subtitle files in a directory and all its subdirectories.
"""

import os
import time
import sys
import gzip
import multiprocessing
import xml.etree.ElementTree

from . import io

__version__ = '0.2.0'

FRAMES_PER_SECOND = 30
PUNCTUATION = tuple(".,:;?!()[]'")


def _parse_time_string(time_string):
    """
    parses string and returns time in seconds.

    """
    # make commas and colons the same symbol and split
    hours, minutes, seconds, frames = time_string.replace(',', ':').split(':')
    return (float(hours) * 60 * 60 +
            float(minutes) * 60 +
            float(seconds) +
            float(frames) / FRAMES_PER_SECOND)


def read_clean_gzfile(gz_file_path, *, break_duration=2.0):
    """
    Generator that opens and reads a gunzipped xml subtitle file, while all
    xml tags and timestamps are removed.

    Parameters
    ----------
    break_duration : float
        defines the amount of time in seconds that need to pass between two
        subtitles in order to start a new paragraph in the resulting corpus.

    Yields
    ------
    line : non empty, cleaned line out of the xml subtitle file

    Raises
    ------
    FileNotFoundError : if file is not there.

    """

    with gzip.open(gz_file_path, "rt", encoding="utf-8-sig") as file_:
        tree = xml.etree.ElementTree.parse(file_)
        root = tree.getroot()

        last_time = 0.0
        for sentence_tag in root.findall('s'):
            # in an s_tag (more or less referring to a 'sentence') there exists
            # time_tags and w_tags (for 'words').

            # join all wordswith spaces in between
            words = []
            for word_tag in sentence_tag.findall('w'):
                text = word_tag.text
                if text in PUNCTUATION:
                    words.append(text)
                elif text is not None:
                    words.extend((' ', text))
                else:
                    raise ValueError("Text content of word tag is None.")
            result = ''.join(words)
            result = result.strip()

            if not result:
                continue

            # Check time and make a new paragraph if needed
            for time_tag in sentence_tag.findall('time'):
                # tag_type is either 'S' or 'E' (start or end)
                tag_type = time_tag.get('id')[-1:]

                current_time = _parse_time_string(time_tag.get('value'))

                # start
                if (tag_type == 'S' and
                        current_time - last_time > break_duration):
                    result = '\n' + result
                # end
                elif tag_type == 'E':
                    last_time = current_time
                elif tag_type == 'S':
                    pass
                else:
                    raise ValueError("tag_type '%s' is not 'S' or 'E'" %
                                     tag_type)

            yield result + "\n"


class JobParseGz():
    # pylint: disable=E0202,missing-docstring

    """
    Stores the persistent information over several jobs and exposes a job
    method that only takes the varying parts as one argument.

    .. note::

        Using a closure is not possible as it is not pickable / serializable.

    """

    def __init__(self, break_duration):
        self.break_duration = break_duration

    def run(self, filename):
        not_found = None
        try:
            lines = list(read_clean_gzfile(filename,
                                           break_duration=self.break_duration))
            lines.append("\n---END.OF.DOCUMENT---\n\n")
        except FileNotFoundError:
            not_found = filename + "\n"
        return (lines, not_found)


def create_corpus_from_gz(directory, outfile, *, n_threads=1, verbose=False):
    """
    Create a corpus file from a set of gunziped (.gz) files in a directory.

    Parameters
    ----------
    directory : str
        use all gz-files in this directory and all subdirectories as input.
    outfile : str
        name of the outfile that will be created.
    n_threads : int
        number of threads to use.
    verbose : bool

    """
    if not os.path.isdir(directory):
        raise OSError("%s does not exist.")
    if os.path.isfile(outfile):
        raise OSError("%s exists. Please <outfile> needs to be new file name."
                      % outfile)

    if verbose:
        print("Walk through '%s' and read in all file names..." % directory)
    gz_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(directory, followlinks=True)
                for name in files
                if name.endswith((".gz",))]
    gz_files.sort()
    if verbose:
        print("Start processing %i files." % len(gz_files))
        start_time = time.time()
    not_founds = list()
    with multiprocessing.Pool(n_threads) as pool:
        with open(outfile, "wt") as result_file:
            progress_counter = 0
            n_files = len(gz_files)
            job = JobParseGz(break_duration=5.0)
            for lines, not_found in pool.imap(job.run, gz_files):
                progress_counter += 1
                if verbose and progress_counter % 1000 == 0:
                    print("%i%% " % (progress_counter / n_files * 100), end="")
                    sys.stdout.flush()

                if lines is not None:
                    result_file.writelines(lines)
                elif not_found is not None:
                    not_founds.append(not_found)
                else:
                    raise NotImplementedError("This should never happend!")
    if verbose:
        duration = time.time() - start_time
        print("\nProcessed %i files. %i files where not found." %
              (len(gz_files), len(not_founds)))
        print("Processing took %.2f seconds (%ih%.2im)." %
              (duration, duration // (60 * 60), duration // 60))

    if not_founds:
        file_name = io.safe_write_path(outfile + ".not_found", template='{path}-{counter}')

        with open(file_name, "wt") as not_found_file:
            not_found_file.writelines(not_founds)
