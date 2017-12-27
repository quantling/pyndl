import os
import sys
import multiprocessing as mp
from pip._vendor import pkg_resources


__author__ = ('David-Elias Künstle, Lennard Schneider, '
              'Konstantin Sering, Marc Weitz')
__author_email__ = 'konstantin.sering@uni-tuebingen.de'
__version__ = '0.3.4'
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


# pylint: disable=W0622
__doc__ = """
:abstract: %s
:version: %s
:author: %s
:contact: %s
:date: 2017-12-27
:copyright: %s
""" % (__description__, __version__, __author__, __author_email__, __license__)


def sysinfo():
    """
    Prints system the dependency information
    """
    pyndl = pkg_resources.working_set.by_key["pyndl"]
    dependencies = [str(r) for r in pyndl.requires()]

    header = "Pyndl " + __version__ + " Information\n"
    header += ("=" * (len(header) - 1)) + "\n"
    header += "\n"

    system, node, kernel, version, machine = os.uname()

    osinfo = "Operating System\n"
    osinfo += ("-" * (len(osinfo) - 1)) + "\n"
    osinfo += "OS: " + system + " " + machine + "\n"
    osinfo += "Kernel: " + kernel + "\n"
    osinfo += "CPU: " + str(mp.cpu_count()) + "\n"
    if system == "Linux":
        names, memory, swap = os.popen("free -m").readlines()
        ix, total, used, *rest = memory.split()
        osinfo += "Memory: " + used + "MiB/" + total + "MiB\n"
        ix, total, used, *rest = swap.split()
        osinfo += "Swap: " + used + "MiB/" + total + "MiB\n"
    osinfo += "\n"

    py = "Python\n"
    py += ("-" * (len(py) - 1)) + "\n"
    py += "Version: " + sys.version.split()[0] + "\n"
    py += "\n"

    deps = "Dependencies\n"
    deps += ("-" * (len(deps) - 1)) + "\n"
    for dep in dependencies:
        pkg = __import__(dep)
        deps += pkg.__name__ + ": " + pkg.__version__ + "\n"

    print(header + osinfo + py + deps)
