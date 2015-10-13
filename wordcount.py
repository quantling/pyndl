#!/usr/bin/env python3

import sys
import os
from collections import Counter
import itertools
import functools
import multiprocessing

def generate_word_list(filename, start, step):
    words = Counter()
    symbols = Counter()
    with open(filename, 'r') as dfile:
        for nn, line in enumerate(itertools.islice(dfile, start, None, step)):
            for word in line.split(' '):
                word = word.strip()
                word = word.strip('!?,.:;/"\'()^@*~')
                word = word.lower()
                if not word:
                    continue
                words[word] += 1
                symbols += Counter(word)
            if nn % 100000 == 0:
                print('.', end='')
                sys.stdout.flush()
    return (words, symbols)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.exit('Usage: python3 %s corpus_file.txt [num_of_processes]' % sys.argv[0])
    if not os.path.exists(sys.argv[1]):
        sys.exit('ERROR: Corpus file %s was not found!' % sys.argv[1])

    filename = sys.argv[1]
    try:
        step = int(sys.argv[2])
    except IndexError:
        step = 1

    def func(start):
        return generate_word_list(filename, start, step)

    with multiprocessing.Pool(step) as pool:
        results = pool.map(func, range(step))
        words = Counter()
        symbols = Counter()
        for words_process, symbols_process in results:
            words += words_process
            symbols += symbols_process

    print('\n...counting done.')

    with open(filename + '.words', 'w') as dfile:
        with open(filename + '.words.freq', 'w') as dfile_freq:
            for word, count in words.most_common():
                dfile.write('{word}\n'.format(word=word))
                dfile_freq.write('{word} {count}\n'.format(count=count, word=word))


    with open(filename + '.symbols', 'w') as dfile:
        dfile.writelines('{symbol} {count}\n'.format(symbol=symbol,
                count=count) for symbol, count in sorted(symbols.items()))

