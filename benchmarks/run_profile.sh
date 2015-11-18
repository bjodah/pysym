#!/bin/bash
source activate se
PYTHONPATH=$(pwd)/.. python -m cProfile -o profile.out profile.py
