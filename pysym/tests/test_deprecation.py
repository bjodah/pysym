import pytest
# import sys

from ..core import Symbol


def test_div():
    x = Symbol('x')
    # if sys.version_info[0] == 2:
    try:
        x/3
    except Warning:
        raise ValueError("got a warning")
    else:
        pass
