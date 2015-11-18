#!/usr/bin/env python
# -*- coding: utf-8 -*-

from benchmarks.Lambdify import TimeLambdifyEvalPySym

instance = TimeLambdifyEvalPySym()
instance.setup(400)
instance.time_evaluate(400)
