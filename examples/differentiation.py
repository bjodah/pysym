#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pysym import Symbol, acos, exp


def main():

    print('Differentiation:')
    x, y = map(Symbol, 'x y'.split())
    expr = (x - acos(y))*exp(x + y)
    Dexpr = expr.diff(y)
    print(Dexpr)
    print(Dexpr._print_ccode())


if __name__ == '__main__':
    main()
