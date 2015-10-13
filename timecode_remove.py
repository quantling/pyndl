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

TIMESTAMP = time.strftime("%Y%m%d_%H%M")
OUTFILE = "../clean.corpus.output.utf8.%s.txt" % TIMESTAMP
FILES_DIR = "xml/hr/"
RECURSIVE = True

if __name__ == "__main__":
    # define rules for substitution
    #<U+FEFF>1
    #myBOMre = re.compile(ur'♬～', re.UNICODE)
    paren = re.compile('\(.*\)', re.UNICODE)
    squ_bracket = re.compile('\[.*\]', re.UNICODE)
    crap = re.compile('"\d\d:.*\d\d"', re.UNICODE)
    music = re.compile('♬～', re.UNICODE)
    arrow = re.compile('➡', re.UNICODE)
    bracket = re.compile('<.*>', re.UNICODE+re.IGNORECASE)

    #files = os.listdir(FILES_DIR)
    gz_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(FILES_DIR)
             for name in files
             if name.endswith((".gz",))]
    gz_files = gz_files[:100]  # test the first 100


    with open(OUTFILE, "wt") as result_file:
        for gzip.filename in gz_files:
            with open(filename, "rt", encoding="utf-8-sig") as file:
                for line in file:
                    line = line.strip().lstrip().rstrip()
                    result = paren.sub('', line)
                    result = squ_bracket.sub('', result)
                    result = music.sub('', result)
                    result = arrow.sub('', result)
                    result = bracket.sub('', result)
                    result = crap.sub('', result)
                    if not result:
                        continue
                    if re.match('([\d]+:[\d]+|[\d]+)', result, re.M|re.I):
                        continue
                    result_file.write(result + "\n")
            print(".", end="")
            sys.stdout.flush()
            result_file.write("---END.OF.DOCUMENT---\n")

    print("Processed %i files." % len(files))
