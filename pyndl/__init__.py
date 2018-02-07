"""
Pyndl - Naive Discriminative Learning in Python
===============================================

*pyndl* is an implementation of Naive Discriminative Learning in Python. It was
created to analyse huge amounts of text file corpora. Especially, it allows to
efficiently apply the Rescorla-Wagner learning rule to these corpora.

"""

import os
import sys
import multiprocessing as mp
from pip._vendor import pkg_resources


__author__ = ('Konstantin Sering, Marc Weitz, '
              'David-Elias Künstle, Lennard Schneider')
__author_email__ = 'konstantin.sering@uni-tuebingen.de'
__version__ = '0.4.1'
__license__ = 'MIT'
__description__ = ('Naive discriminative learning implements learning and '
                   'classification models based on the Rescorla-Wagner '
                   'equations.')
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    ]


def sysinfo():
    """
    Prints system the dependency information
    """
    pyndl = pkg_resources.working_set.by_key["pyndl"]
    dependencies = [str(r) for r in pyndl.requires()]

    header = ("Pyndl Information\n"
              "=================\n\n")

    general = ("General Information\n"
               "-------------------\n"
               "Python version: {}\n"
               "Pyndl version: {}\n\n").format(sys.version.split()[0], __version__)

    uname = os.uname()
    osinfo = ("Operating System\n"
              "----------------\n"
              "OS: {s.sysname} {s.machine}\n"
              "Kernel: {s.release}\n"
              "CPU: {cpu_count}\n").format(s=uname, cpu_count=mp.cpu_count())

    if uname.sysname == "Linux":
        names, *lines = os.popen("free -m").readlines()
        for identifier in ["Mem:", "Swap:"]:
            memory = [line for line in lines if identifier in line][0]
            ix, total, used, *rest = memory.split()
            osinfo += "{} {}MiB/{}MiB\n".format(identifier, used, total)

    osinfo += "\n"

    deps = ("Dependencies\n"
            "------------\n")

    deps += "\n".join("{pkg.__name__}: {pkg.__version__}".format(pkg=__import__(dep))
                      for dep in dependencies)

    print(header + general + osinfo + deps)
