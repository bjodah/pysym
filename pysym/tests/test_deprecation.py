import pytest
# import sys

from ..core import Symbol


def test_div():
    x = Symbol('x')
    # if sys.version_info[0] == 2:
    with pytest.warns(None):
        x/3
