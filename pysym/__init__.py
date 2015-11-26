# -*- coding: utf-8 -*-
"""
pysym is a minimal symbolic manipulation framework
"""

from __future__ import (absolute_import, division, print_function)

from ._release import __version__
import os


if os.environ.get('PYSYM_USE_NATIVE', '0') == '1':
    # Will eventually provide a faster alternative
    from ._pysym import (
        Symbol, Dummy, Number, ITE, gamma, abs, exp, log,
        sin, cos, tan, asin, acos, atan,
        Vector, Matrix, sqrt, _wrap_numbers, Add, Mul,
        Lt, Le, Eq, Ne, Gt, Ge
    )
elif os.environ.get('PYSYM_USE_SYMPY', '0') == '1':
    # For debugging purposes only
    from sympy import (
        Symbol, Dummy, Number, ITE, gamma, abs, exp, log,
        sin, cos, tan, asin, acos, atan,
        Vector, Matrix, sqrt, _wrap_numbers, Add, Mul,
        Lt, Le, Eq, Ne, Gt, Ge
    )
else:
    from .core import (
        Symbol, Dummy, Number, ITE, gamma, abs, exp, log,
        sin, cos, tan, asin, acos, atan,
        Vector, Matrix, sqrt, _wrap_numbers, Add, Mul,
        Lt, Le, Eq, Ne, Gt, Ge
    )

from .util import lambdify, Lambdify, symbols, symarray  # noqa


pi = 4*atan(1)
