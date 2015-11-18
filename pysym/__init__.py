# -*- coding: utf-8 -*-
"""
pysym is a minimal symbolic manipulation framework
"""

from __future__ import (absolute_import, division, print_function)

from ._release import __version__
from .core import (
    Symbol, Number, ITE, gamma, abs, exp, log, sin, cos, tan, asin, acos, atan,
    Vector, Matrix, sqrt
)
from .util import lambdify, Lambdify, symbols, symarray


pi = 4*atan(1)
