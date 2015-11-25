#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from benchmarks.Lambdify import TimeLambdifyEvalPySym


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    instance = TimeLambdifyEvalPySym()
    instance.setup(n)
    instance.time_evaluate(n)
