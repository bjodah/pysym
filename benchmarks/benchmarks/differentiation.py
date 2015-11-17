
import functools
import operator

import pysym


class Diff:

    def setup(self):
        x, y, z = self.symbols = map(pysym.Symbol, 'x y z'.split())
        self.expr = functools.reduce(operator.add, [x**i/(y**i - i/z) for i in range(3)])
        self.y0 = [.1, .2, 0, 0]

    def time_integrate_scipy(self):
        self.odesys.integrate('scipy', self.tout, self.y0)

    def time_integrate_gsl(self):
        self.odesys.integrate('gsl', self.tout, self.y0)

    def time_integrate_odeint(self):
        self.odesys.integrate('odeint', self.tout, self.y0)

    def time_integrate_cvode(self):
        self.odesys.integrate('cvode', self.tout, self.y0)
