#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
corpus.py generates a corpus file (outfile) out of a bunch of gunzipped xml
subtitle files in a directory and all its subdirectories.

Usage:
    corpus.py [-n=N_THREADS] [-v] <directory> <outfile>
    corpus.py -h | --help
    corpus.py --version

Options:
    -h, --help      Show this screen.
    -n=N_THREADS    Number of threads to use [default: 1].
    -v              Verbose output.

"""

__version__ = '0.1.0'

import fileinput
import re
import os
import time
import sys
import gzip
import multiprocessing
import xml.etree.ElementTree

if __name__ == '__main__':
    from docopt import docopt
else:
    from .docopt import docopt

def read_clean_gzfile(gz_file_path):
    """
    Generator that opens and reads a gunzipped xml subtitle file, while all
    xml tags and timestamps are removed.

    Yields
    ======
    line : non empty, cleaned line out of the xml subtitle file

    Raises
    ======
    FileNotFoundError : if file is not there.

    """
    
    # The time_threshold defines the amount of time to pass in order
    # to start a new paragraph (this might have to be a parameter of the function!)
    current_time = 0
    time_threshold = 10
    
    with gzip.open(gz_file_path, "rt", encoding="utf-8-sig") as file_:
        tree = xml.etree.ElementTree.parse(file_)
        root = tree.getroot()
	  
        for s_tag in root.findall('s'):
            # inside s tags we can find time tags (self-explanatory) and w tags (which contain words)
            
            # read all words in w tags and concatenate 
            result = ""
            for w_tag in s_tag.findall('w'):
                result += w_tag.text + " "
            
            if not result:
                continue
                
            # Check time and prepend a newspace (new paragraph) if needed
            for time_tag in s_tag.findall('time'):
	        # tag_type is either 'S' or 'E'
                tag_type = time_tag.get('id')[-1:]
                
                # parse time value to seconds
                t_string = time_tag.get('value').replace(',',':').split(':')
                t = float(t_string[0])*(60*60) + float(t_string[1])*60 + float(t_string[2]) + float(t_string[3])/100
                
                if (tag_type == 'S' and t-current_time > time_threshold):
                    result = "\n" + result
                elif (tag_type == 'E'):
                    current_time = t
                
            yield result + "\n"


def _job(filename):
    """Job for threads in multiprocessing."""
    lines = None
    not_found = None
    try:
        lines = list(read_clean_gzfile(filename))
        lines.append("\n---END.OF.DOCUMENT---\n\n")
    except FileNotFoundError:
        not_found = filename + "\n"
    return (lines, not_found)


def main(directory, outfile, *, n_threads=1, verbose=False):
    """
    Mein corpus.py program that starts the multiple processes and collects all
    the results.

    Parameters
    ==========
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
             for root, dirs, files in os.walk(directory)
             for name in files
             if name.endswith((".gz",))]

    if verbose:
        print("Start processing %i files." % len(gz_files))
        start_time = time.time()
    not_founds = list()
    with multiprocessing.Pool(n_threads) as pool:
        with open(outfile, "wt") as result_file:
            progress_counter = 0
            n_files = len(gz_files)
            for lines, not_found in pool.imap_unordered(_job, gz_files):

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
        print("Processing took %.2f seconds (%ih%.2im)." % (duration, duration //
                                                          (60 * 60), duration // 60) )

    if not_founds:
        # prevent overwriting files
        file_name = outfile + ".not_found"
        nn = 1
        while True:
            if os.path.isfile(file_name):
                nn += 1
                file_name = outfile + ".not_found-" + str(nn)
                continue
            else:
                break

        with open(file_name, "wt") as not_found_file:
            not_found_file.writelines(not_founds)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='corpus %s' % __version__)
    main(arguments['<directory>'],
         arguments['<outfile>'],
         n_threads=int(arguments['-n']),
         verbose=arguments['-v'])

