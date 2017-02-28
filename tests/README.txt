This file describes the used test resources.

corpus.txt
----------
The corpus file is based on the print version of the wikipedia article about the Rescorla–Wagner-modell (2017-02-28, https://hu.wikipedia.org/wiki/Rescorla–Wagner-modell, CC BY-SA 3.0) in Hungarian.
HTML tags were removed mostly automatically with `pandoc --from=html --to=plain corpus.html`, in addition by hand.
The remaining words were separated at whitespace to lines by `tr -s '[[:space:]]' '\n'`.
