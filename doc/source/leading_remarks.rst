Leading Remarks
===============

Important terminology
---------------------
Some terminology used in these modules and scripts which describe specific
behaviour:

cue :
    A cue is something that gives a hint on something else. The something else
    is called outcome. Examples for cues in a text corpus are trigraphs or
    preceding words for the word or meaning of the word.

outcome :
    The outcome is the result of an event. Examples are words, the meaning of
    the word, or lexomes.

event :
    An event connects cues with outcomes. In any event one or more unordered
    cues are present and one or more outcomes are present.

weights :
    The weights represent the learned experience / association between all cues
    and outcomes of interest. Usually, some meta data is stored alongside the
    learned weights.

cue file :
    A cue file contains a list of all cues that are interesting for a specific
    question. It is a utf-8 encoded tab delimitered text file with a
    header in the first line. It has two columns. The first column contains the
    cue and the second column contains the frequency of the cue. There is one
    cue per line. The ordering does not matter.

outcome file :
    An outcome file contains a list of all outcomes that are interesting for a
    specific question. It is a utf-8 encoded tab delimitered text file with a
    header in the first line. It has two columns. The first column contains the
    outcome and the second column contains the frequency of the outcome. There
    is one outcome per line. The ordering does not matter.

symbol file :
    A symbol file contains a list of all symbols that are allowed to occur in
    an outcome or a cue. It is a utf-8 encoded tab delimitered text file with a
    header in the first line. It has two columns. The first column contains the
    symbol and the second column contains the frequency of the symbol. There is
    one symbol per line. The ordering does not matter.

event file :
    An event file contains a list of all events that should be learned. The
    learning will start at the first event and continue to the last event in
    order of the lines. The event file is a utf-8 encoded tab delimitered
    gzipped text file with a header in the first line. It has three columns.
    The first column contains an underscore delimitered list of all cues. The
    second column contains an underscore delimitered list of all outcomes. The
    third column contains the frequency of the event. The ordering of the cues
    and outcomes does not matter. There is one event per line. The ordering of
    the lines in the file *does* matter. In ``pyndl`` the event file will be
    temporally stored into a binary format to speed up computations.

corpus file :
    A corpus file is a utf-8 encoded text file that contains huge amounts of
    text. A ``---end.of.document---`` or ``---END.OF.DOCUMENT---`` string marks
    where an old document finished and a new document starts.

weights file :
    A weights file contains the learned weights between cues and outcomes. The
    netCDF format is used to store these information along side with meta data,
    which contains the learning parameters, the time needed to calculate the
    weights, the version of the software used and other information.

