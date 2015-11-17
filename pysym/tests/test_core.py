# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import functools
import operator

from ..core import Symbol, Add, gamma, Number, sin, cos, Mul, ITE


def test_Symbol():
    s1 = Symbol('s')
    s2 = Symbol('s')
    assert s1 is s2

def test_pow():
    s = Symbol('s')
    assert (s**1).has(s)
    assert (s**0).evalf() == 1
    Zero = Number(0)
    One = Number(1)
    assert Zero**Zero is One

def test_Symbol_add():
    x, y = map(Symbol, 'x y'.split())
    xy = x + y
    assert isinstance(xy, Add)
    assert len(xy.args) == 2
    xyy = x + y + y
    assert isinstance(xyy, Add)
    assert len(xyy.args) == 2


def test_gamma():
    five = Number(5)
    f = gamma(five)
    assert abs(f.evalf() - 4*3*2) < 1e-15


def test_Add():
    one = Number(1)
    two = Number(2.0)
    one_plus_two = one + two
    assert abs(one_plus_two.evalf() - 3) < 1e-15


def test_division():
    x, y = map(Symbol, 'x y'.split())
    assert (0/x == 0/x).evalb()

    expr = x/y
    assert abs(expr.subs({x: Number(3), y: Number(7)}).evalf() - 3/7) < 1e-15

    assert (x/3 == x/3).evalb()
    assert (3/y == 3/y).evalb()
    assert (3/x != 3/y).evalb()


def test_addition_subs():
    x, y = map(Symbol, 'x y'.split())
    expr = x + y
    assert abs(expr.subs({x: Number(3), y: Number(7)}).evalf() - 10) < 1e-15


def test_subtraction():
    x, y = map(Symbol, 'x y'.split())
    expr1 = x - y
    expr2 = expr1.subs({x: Number(3), y: Number(7)})
    assert abs(expr2.evalf() + 4) < 1e-15


def test_is_atomic():
    x = Symbol('x')
    assert x.is_atomic
    assert not sin(x).is_atomic


def test_eq_evalb():
    x = Symbol('x')
    assert (sin(x) == sin(x)).evalb() is True
    assert (x == x).evalb() is True
    assert (x == sin(x)).evalb() is False


def test_diff1():
    x = Symbol('x')
    sinx = sin(x)
    assert (sinx.diff(x) == cos(x)).evalb()
    assert not (sinx.diff(x) == sin(x)).evalb()

    assert ((0/(1+x)).diff(x) == (0/(1+x)).diff(x)).evalb()
    assert ((0/(1-x)).diff(x) == (0/(1-x)).diff(x)).evalb()
    f = x**0/(2 - 1*(0/x))
    dfdx = f.diff(x)
    print(dfdx)
    assert dfdx.evalf() == 0

def test_subs():
    x, y = map(Symbol, 'x y'.split())

    x_plus_y = x + y
    x_plus_y = x_plus_y.subs({x: Number(3)})
    x_plus_y = x_plus_y.subs({y: Number(7)})
    assert abs(x_plus_y.evalf() - 10) < 1e-15


def test_diff2():
    x, y = map(Symbol, 'x y'.split())
    assert x.diff(x) == 1
    assert y.diff(x) == 0

    x5 = x*x*x*x*x
    assert abs(x5.diff(x).subs({x: Number(7)}).evalf() - 5*7**4) < 4e-12


def test_repr():
    x, y = map(Symbol, 'x y'.split())
    assert repr(x + y) == "Add(Symbol('x'), Symbol('y'))"


def test_str():
    x, y = map(Symbol, 'x y'.split())
    assert str(x + y) == '(x + y)'


def test_sorting():
    x, y = map(Symbol, 'x y'.split())
    a = Add(x, x, x, y, x, x)
    s = a.sorted()
    assert s == Add(x, x, x, x, x, y)


def test_collecting():
    x, y = map(Symbol, 'x y'.split())
    a = Add(x, x, x, y, x, x)
    assert a == Add(Mul(Number(5), x), y)


def test_Number_mul():
    n = Number(3)
    n2 = n*n
    assert abs(n2.evalf() - 9) < 1e-15


def test_neg():
    m1 = -Number(1)
    assert abs(m1.evalf() + 1) < 1e-15

    m2 = m1*m1
    assert abs(m2.evalf() - 1) < 1e-15


def test_ITE():
    x, y = map(Symbol, 'x y'.split())
    r = x < y
    x1y2 = {x: Number(1), y: Number(2)}
    x1y1 = {x: Number(1), y: Number(1)}
    x1y0 = {x: Number(1), y: Number(0)}
    assert r.subs(x1y2).evalb() is True
    assert r.subs(x1y1).evalb() is False
    assert r.subs(x1y0).evalb() is False

    ite = ITE(x < y, Number(3), Number(7))
    assert abs(ite.subs(x1y2).evalf() - 3) < 1e-15
    assert abs(ite.subs(x1y0).evalf() - 7) < 1e-15


def test_diff3():
    x, y, z = map(Symbol, 'x y z'.split())
    f = functools.reduce(operator.add,
                         [x**i/(y**i - i/z) for i in range(2)])
    dfdx = f.diff(x)
