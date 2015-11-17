import pytest

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
