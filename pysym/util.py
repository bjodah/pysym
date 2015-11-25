# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)
from . import _wrap_numbers, Symbol, Number, Matrix


def symbols(s):
    """ mimics sympy.symbols """
    tup = tuple(map(Symbol, s.replace(',', ' ').split()))
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def symarray(prefix, shape):
    import numpy as np
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = Symbol('%s_%s' % (
            prefix, '_'.join(map(str, index))))
    return arr


def lambdify(args, exprs):
    """
    lambdify mimics sympy.lambdify
    """
    try:
        nargs = len(args)
    except TypeError:
        args = (args,)
        nargs = 1
    try:
        nexprs = len(exprs)
    except TypeError:
        exprs = (exprs,)
        nexprs = 1

    @_wrap_numbers
    def f(*inp):
        if len(inp) != nargs:
            raise TypeError("Incorrect number of arguments")
        try:
            len(inp)
        except TypeError:
            inp = (inp,)
        subsd = dict(zip(args, inp))
        return [expr.subs(subsd).evalf() for expr in exprs][
            0 if nexprs == 1 else slice(None)]
    return f


class Lambdify(object):
    """
    Lambdify mimics symengine.Lambdify
    """

    def __init__(self, syms, exprs):
        self.syms = syms
        self.exprs = exprs

    def __call__(self, inp, out=None):
        inp = tuple(map(Number.make, inp))
        subsd = dict(zip(self.syms, inp))

        def _eval(expr_iter):
            return [expr.subs(subsd).evalf() for expr in expr_iter]
        exprs = self.exprs
        if out is not None:
            try:
                out.flat = _eval(exprs.flatten())
            except AttributeError:
                out.flat = _eval(exprs)
        elif isinstance(exprs, Matrix):
            import numpy as np
            nr, nc = exprs.nrows, exprs.ncols
            out = np.empty((nr, nc))
            for ri in range(nr):
                for ci in range(nc):
                    out[ri, ci] = exprs._get_element(
                        ri*nc + ci).subs(subsd).evalf()
            return out
            # return Matrix(nr, nc, _eval(exprs._get_element(i) for
            #                             i in range(nr*nc)))
        elif hasattr(exprs, 'reshape'):
            # NumPy like container:
            container = exprs.__class__(exprs.shape, dtype=float, order='C')
            container.flat = _eval(exprs.flatten())
            return container
        else:
            return _eval(exprs)
