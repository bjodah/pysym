# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)

import array
import math
import itertools
import pytest

# Tests I wrote for Lambdify in symengine reference 'se'
import pysym as se

from .. import (lambdify, Symbol, sqrt, sin, cos, pi)


s0, s1, s2 = map(Symbol, 's0,s1,s2'.split(','))


def test_single_arg():
    f = lambdify(s0, 2*s0)
    assert f(1) == 2


def test_list_args():
    f = lambdify([s0, s1], s0 + s1)
    assert f(1, 2) == 3


def test_sin():
    f = lambdify(s0, sin(s0))
    assert f(0) == 0.


def test_exponentiation():
    f = lambdify(s0, s0**2)
    assert f(-1) == 1
    assert f(0) == 0
    assert f(1) == 1
    assert f(-2) == 4
    assert f(2) == 4
    assert f(2.5) == 6.25


def test_sqrt():
    f = lambdify(s0, sqrt(s0))
    assert f(0) == 0.
    assert f(1) == 1.
    assert f(4) == 2.
    assert abs(f(2) - 1.414) < 0.001
    assert f(6.25) == 2.5


def test_trig():
    f = lambdify([s0], [cos(s0), sin(s0)])
    d = f(pi)
    prec = 1e-11
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec
    d = f(3.14159)
    prec = 1e-5
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec


def allclose(iter_a, iter_b, rtol=1e-10, atol=1e-10):
    for a, b in zip(iter_a, iter_b):
        if (abs(a-b) < abs(a)*rtol + atol) is not True:
            return False
    return True


def test_vector_simple():
    f = lambdify((s0, s1, s2), (s2, s1, s0))
    assert allclose(f(3, 2, 1), (1, 2, 3))
    assert allclose(f(1., 2., 3.), (3., 2., 1.))
    with pytest.raises(TypeError):
        f(0)


def test_trig_symbolic():
    f = lambdify([s0], [cos(s0), sin(s0)])
    d = f(pi)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_trig_float():
    f = lambdify([s0], [cos(s0), sin(s0)])
    d = f(3.14159)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_Lambdify():
    n = 7
    args = x, y, z = se.symbols('x y z')
    l = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z])
    assert allclose(l(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])


def _get_2_to_2by2_numpy():
    import numpy as np
    args = x, y = se.symbols('x y')
    exprs = np.array([[x+y+1.0, x*y],
                      [x/y, x**y]])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X+Y+1.0)) < 1e-15
        assert abs(A[0, 1] - (X*Y)) < 1e-15
        assert abs(A[1, 0] - (X/Y)) < 1e-15
        assert abs(A[1, 1] - (X**Y)) < 1e-13
    return l, check


def test_Lambdify_2dim_numpy():
    import numpy as np
    lmb, check = _get_2_to_2by2_numpy()
    for inp in [(5, 7), np.array([5, 7]), [5.0, 7.0]]:
        A = lmb(inp)
        assert A.shape == (2, 2)
        check(A, inp)


def _get_array():
    X, Y, Z = inp = array.array('d', [1, 2, 3])
    args = x, y, z = se.symbols('x y z')
    exprs = [x+y+z, se.sin(x)*se.log(y)*se.exp(z)]
    ref = [X+Y+Z, math.sin(X)*math.log(Y)*math.exp(Z)]

    def check(arr):
        assert all([abs(x1-x2) < 1e-13 for x1, x2 in zip(ref, arr)])
    return args, exprs, inp, check


def test_array():
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out = lmb(inp)
    check(out)


def _get_1_to_2by3_matrix():
    x = se.symbols('x')
    args = x,
    exprs = se.Matrix(2, 3, [x+1, x+2, x+3,
                             1/x, 1/(x*x), 1/(x**3.0)])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, = inp
        assert abs(A[0, 0] - (X+1)) < 1e-15
        assert abs(A[0, 1] - (X+2)) < 1e-15
        assert abs(A[0, 2] - (X+3)) < 1e-15
        assert abs(A[1, 0] - (1/X)) < 1e-15
        assert abs(A[1, 1] - (1/(X*X))) < 1e-15
        assert abs(A[1, 2] - (1/(X**3.0))) < 1e-15
    return l, check


def test_2dim_Matrix():
    l, check = _get_1_to_2by3_matrix()
    inp = [7]
    check(l(inp), inp)


def test_jacobian():
    import numpy as np
    x, y = se.symbols('x, y')
    args = se.Matrix(2, 1, [x, y])
    v = se.Matrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((2, 2))
    inp = X, Y = 7, 11
    lmb(inp, out)
    assert np.atleast_1d(out).shape == (2, 2)
    assert np.allclose(out, [[3 * X**2 * Y, X**3],
                             [Y + 1, X + 1]])


def test_jacobian2():
    import numpy as np
    x, y = se.symbols('x, y')
    args = se.Matrix(2, 1, [x, y])
    v = se.Matrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((2, 2))
    inp = X, Y = 7, 11
    out = lmb(inp)
    assert np.atleast_1d(out).shape == (2, 2)
    assert np.allclose(out, [[3 * X**2 * Y, X**3],
                             [Y + 1, X + 1]])


def test_itertools_chain():
    args, exprs, inp, check = _get_array()
    l = se.Lambdify(args, exprs)
    inp = itertools.chain([inp[0]], (inp[1],), [inp[2]])
    A = l(inp)
    check(A)


def test_Lambdify_matrix():
    import numpy as np
    x, y = arr = se.symarray('x', 2)
    mat = se.Matrix(2, 2, [x, 1+y, 2*y*x**2, 3])
    lmb = se.Lambdify(arr, mat)
    result = lmb([3, 5])
    assert result.shape == (2, 2)
    assert np.allclose(result, [[3, 6], [90, 3]])
