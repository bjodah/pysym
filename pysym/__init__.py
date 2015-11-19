# -*- coding: utf-8 -*-
"""
pysym is a minimal symbolic manipulation framework
"""

from __future__ import (absolute_import, division, print_function)

from ._release import __version__
import os

if os.environ.get('PYSYM_USE_NATIVE', '0') == '1':
    from ._pysym import (
        Symbol, Number, ITE, gamma, abs, exp, log, sin, cos, tan, asin, acos, atan,
        Vector, Matrix, sqrt, _wrap_numbers, Add, Mul,
        Lt, Le, Eq, Ne, Gt, Ge
    )
else:
    from .core import (
        Symbol, Number, ITE, gamma, abs, exp, log, sin, cos, tan, asin, acos, atan,
        Vector, Matrix, sqrt, _wrap_numbers, Add, Mul,
        Lt, Le, Eq, Ne, Gt, Ge
    )

from .util import lambdify, Lambdify, symbols, symarray


pi = 4*atan(1)
