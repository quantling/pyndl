#!/usr/bin/env python3

# pylint: disable=C0111

import re
from io import StringIO
from contextlib import redirect_stdout

import pyndl


def test_sysinfo():
    out = StringIO()
    with redirect_stdout(out):
        pyndl.sysinfo()
    out = out.getvalue()

    pattern = re.compile(r"[a-zA-Z0-9_\. ]*\n[\=]*\n+([a-zA-Z0-9_ ]*\n[\-]*\n"
                         r"([a-zA-Z0-9_ ]*: [a-zA-Z0-9_\.\-/ ]*\n+)+)+")
    assert pattern.match(out)
