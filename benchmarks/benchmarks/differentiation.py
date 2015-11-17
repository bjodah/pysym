
import functools
import operator

import pysym


class Diff:

    def setup(self):
        x, y, z = self.symbols = map(pysym.Symbol, 'x y z'.split())
        self.expr = functools.reduce(operator.add, [x**i/(y**i - i/z) for i in range(3)])
        self.y0 = [.1, .2, 0, 0]

    def time_diff_x(self):
        self.expr.diff(self.symbols[0])

    def time_diff_y(self):
        self.expr.diff(self.symbols[1])

    def time_diff_z(self):
        self.expr.diff(self.symbols[2])
