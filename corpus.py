#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# By Cyrus Shaoul
# Modified by Tino Sering

import fileinput
import re
import os
import time
import sys
import gzip
import multiprocessing

TIMESTAMP = time.strftime("%Y%m%d_%H%M")
OUTFILE = "pl.corpus.utf8.%s.txt" % TIMESTAMP
FILES_DIR = "/home/petar/Corpora/OpenSubtitles2013/xml/pl/"
N_PROC = 16

# define rules for substitution
PATTERNS = (
        # remove xml tags
        re.compile('<.*?>', re.UNICODE+re.IGNORECASE),
        # remove everything within paranthesis
        re.compile('\(.*?\)', re.UNICODE),
        # remove everything within square brackets
        re.compile('\[.*?\]', re.UNICODE),
        )


def read_clean_gzfile(gz_file_path):
    """
    Generator that opens and reads a gunzipped xml subtitle file, while all
    xml tags and timestamps are removed.

    Yields
    ------
    line : non empty, cleaned line out of the xml subtilte file

    Raises
    ------
    FileNotFoundError : if file is not there.

    """
    with gzip.open(gz_file_path, "rt", encoding="utf-8-sig") as file_:
        for line in file_:
            result = line.strip().lstrip().rstrip()
            for pattern in PATTERNS:
                result = pattern.sub('', result)
            if not result:
                continue
            yield result + "\n"


def job(filename):
    lines = None
    not_found = None
    try:
        lines = list(read_clean_gzfile(filename))
        lines.append("---END.OF.DOCUMENT---\n")
    except FileNotFoundError:
        not_found = filename + "\n"
    return (lines, not_found)


def main():
    gz_files = (os.path.join(root, name)
             for root, dirs, files in os.walk(FILES_DIR)
             for name in files
             if name.endswith((".gz",)))

    print("Start processing %i files." % len(gz_files))
    not_founds = list()
    with multiprocessing.Pool(N_PROC) as pool:
        with open(OUTFILE, "wt") as result_file:
            progress_counter = 0
            n_files = len(gz_files)
            for lines, not_found in pool.imap_unordered(job, gz_files):

                progress_counter += 1
                if progress_counter % 1000 == 0:
                    print("%i%% " % (progress_counter / n_files * 100),
                            end="")
                    sys.stdout.flush()

                if lines is not None:
                    result_file.writelines(lines)
                elif not_found is not None:
                    not_founds.append(not_found)
                else:
                    raise NotImplementedError("This should never happend!")
    print("\nProcessed %i files. %i files where not found." % (len(gz_files),
                                                             len(not_founds)))

    with open("not_found." + OUTFILE, "wt") as not_found_file:
        not_found_file.writelines(not_founds)


if __name__ == "__main__":
    main()

