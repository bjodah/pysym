# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import functools
import math
import operator

from .. import (
    Symbol, Add, gamma, Number, sin, cos, Mul, ITE, exp, Lt, tan, log,
    Abs
)


def test_has():
    x, y = map(Symbol, 'x y'.split())
    summation = (x+1)
    assert summation.has(x)
    assert not summation.has(y)


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


def test_Sub():
    x, y = map(Symbol, 'x y'.split())
    xmy = x - y
    ymx = y - x
    subsd = {x: Number(3.0), y: Number(7.0)}
    assert abs(xmy.subs(subsd).evalf() + 4) < 1e-16
    assert abs(ymx.subs(subsd).evalf() - 4) < 1e-16
    assert abs((xmy + ymx).subs(subsd).evalf()) < 1e-16

    one_minus_x = 1 - x
    assert abs(one_minus_x.subs({x: Number(0.5)}) - 0.5) < 1e-16


def test_division():
    x, y = map(Symbol, 'x y'.split())
    zero_over_x_1 = 0/x
    zero_over_x_2 = 0/x
    assert zero_over_x_1 == zero_over_x_2

    expr = x/y
    assert abs(expr.subs({x: Number(3), y: Number(7)}).evalf() - 3/7) < 1e-15

    assert x/3 == x/3
    assert 3/y == 3/y
    assert 3/x != 3/y


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
    assert x.is_atomic()
    assert not sin(x).is_atomic()


def test_eq_evalb():
    x = Symbol('x')
    assert (sin(x) == sin(x)) is True
    assert (x == x) is True
    assert (x == sin(x)) is False


def test_diff0():
    x = Symbol('x')
    assert ((3*x).diff(x) == Number(3)) is True


def test_diff1():
    x = Symbol('x')
    sinx = sin(x)
    assert sinx.diff(x) == cos(x)
    assert not sinx.diff(x) == sin(x)

    assert (0/(1+x)).diff(x) == (0/(1+x)).diff(x)
    assert (0/(1-x)).diff(x) == (0/(1-x)).diff(x)
    f = x**0/(2 - 1*(0/x))
    dfdx = f.diff(x)
    assert dfdx.evalf() == 0


def test_subs1():
    x, y = map(Symbol, 'x y'.split())
    assert abs((x/y).subs({x: Number(7), y: Number(3)}).evalf() - 7./3) < 1e-15


def test_subs2():
    x, y = map(Symbol, 'x y'.split())

    x_plus_y = x + y
    x_plus_y = x_plus_y.subs({x: Number(3)})
    x_plus_y = x_plus_y.subs({y: Number(7)})
    assert abs(x_plus_y.evalf() - 10) < 1e-15


def test_diff2():
    x, y = map(Symbol, 'x y'.split())
    assert (x.diff(x) == 1) is True
    assert (y.diff(x) == 0) is True

    x5 = x*x*x*x*x
    assert abs(x5.diff(x).subs({x: Number(7)}).evalf() - 5*7**4) < 4e-12


def test_diff3():
    x, y, z = map(Symbol, 'x y z'.split())
    f = functools.reduce(operator.add,
                         [x**i/(y**i - i/z) for i in range(2)])
    dfdx = f.diff(x)
    assert dfdx.has(y)


def test_diff4():
    x, y = map(Symbol, 'x y'.split())
    assert ((3*x).diff(y) == Number(0)) is True


def test_diff5():
    x, y = map(Symbol, 'x y'.split())
    assert ((x**3).diff(y) == Number(0)) is True
    assert ((x**3).diff(x) == 3*x**2) is True
    assert (((2*x+y)**3).diff(x) == 3*(2*x+y)**2*2) is True
    assert abs((x**x).diff(x).subs({x: Number(3.14)}).evalf() -
               (3.14**3.14 * math.log(3.14) + 3.14**3.14)) < 1e-13


def test_diff6():
    x, y = map(Symbol, 'x y'.split())
    expr = x*y**2 - tan(2*x)
    subsd = {x: Number(0.2), y: Number(0.3)}
    ref_val = -2.26750821162195
    assert abs(expr.diff(x).subs(subsd).evalf() - ref_val) < 1e-14


def test_diff7():
    x = Symbol('x')
    expr = 2*exp(x*x)*x
    subsd = {x: Number(0.2)}
    ref_val = 2.2481512722555586
    assert abs(expr.diff(x).subs(subsd).evalf() - ref_val) < 1e-15


def test_diff8():
    x = Symbol('x')
    e = log(sin(x))
    subsd = {x: Number(0.2)}
    ref_val = 4.93315487558689
    assert abs(e.diff(x).subs(subsd).evalf() - ref_val) < 4e-15


def test_diff9():
    x = Symbol('x')
    absx = Abs(x)
    assert abs(absx.diff(x).subs({x: Number(-.1)}).evalf() + 1) < 1e-16
    assert abs(absx.diff(x).subs({x: Number(.1)}).evalf() - 1) < 1e-16
    assert math.isnan(absx.diff(x).subs({x: Number(0)}).evalf())


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
    assert s == Add.create((x, x, x, x, x, y))


def test_collecting():
    x, y = map(Symbol, 'x y'.split())
    a = Add.create((x, x, x, y, x, x))
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
    r = Lt(x, y)  # x < y
    x1y2 = {x: Number(1), y: Number(2)}
    x1y1 = {x: Number(1), y: Number(1)}
    x1y0 = {x: Number(1), y: Number(0)}
    assert r.subs(x1y2).evalb() is True
    assert r.subs(x1y1).evalb() is False
    assert r.subs(x1y0).evalb() is False

    ite = ITE(Lt(x, y), Number(3), Number(7))
    assert abs(ite.subs(x1y2).evalf() - 3) < 1e-15
    assert abs(ite.subs(x1y0).evalf() - 7) < 1e-15


def test_subs_multi():
    s1, s2 = map(Symbol, 's1 s2'.split())
    subsd = {s1: Number(1), s2: Number(1)}
    one_a = s1.subs(subsd)
    assert abs((one_a - 1).evalf()) < 1e-16
    one_b = one_a.subs(subsd)
    assert abs((one_b - 1).evalf()) < 1e-16


def _equal(expr1, expr2):
    return (expr1.sorted() == expr2.sorted()) is True


def test_denest():
    x, y = Symbol('x'), Symbol('y')
    expr = ((x + 1)*y + 1).expand()
    ref = x*y + y + 1
    assert _equal(expr, ref)


def test_expand():
    x, y = Symbol('x'), Symbol('y')
    expr1 = ((x+1)*y).expand()
    ref1 = x*y + 1*y
    assert _equal(expr1, ref1)

    expr2 = ((x**2 + 1)*(y + 2)).expand()
    ref2 = x**2 * y + x**2 * 2 + 1*y + 1*2

    assert _equal(expr2, ref2)

    expr3 = ((x**2 + 1)*(y**2 + y + 2)).expand()
    ref3 = x**2 * y**2 + x**2 * y + x**2 * 2 + y**2 + y + 2
    assert _equal(expr3, ref3)


def test_subs_num():
    ref = 18.901100113049495
    x = list(map(Symbol, ['x_'+str(i) for i in range(14)]))
    p = list(map(Symbol, ['p_'+str(i) for i in range(14)]))
    expr = (-p[1] - 2*p[11] - 3*p[12] - 4*p[13] - p[4] + exp(x[1]) +
            2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + exp(x[4]))
    # make this work without Number.make:
    subsd = dict(zip(x+p, tuple(map(Number.make, [1]*28))))
    val = expr.subs(subsd).evalf()
    assert abs(val - ref) < 1e-14


def test_Pow():
    x = Symbol('x')
    p = x**Number(1)
    assert p is x
