#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from setuptools import setup


pkg_name = 'pysym'

PYSYM_RELEASE_VERSION = os.environ.get('PYSYM_RELEASE_VERSION', '')  # v*
release_py_path = os.path.join(pkg_name, '_release.py')

if len(PYSYM_RELEASE_VERSION) > 0:
    if PYSYM_RELEASE_VERSION[0] == 'v':
        TAGGED_RELEASE = True
        __version__ = PYSYM_RELEASE_VERSION[1:]
    else:
        raise ValueError("Ill formated version")
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(open(release_py_path).read())


classifiers = [
    "Development Status :: 3 - Alpha",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

tests = [
    'pysym.tests',
]

with open(os.path.join(pkg_name, '__init__.py')) as f:
    long_description = f.read().split('"""')[1]
descr = 'Minimal symbolic manipulation framework.'
setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=descr,
    long_description=long_description,
    classifiers=classifiers,
    author='Björn Dahlgren',
    author_email='bjodah@DELETEMEgmail.com',
    url='https://github.com/bjodah/' + pkg_name,
    license='BSD',
    packages=[pkg_name] + tests,
)

if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            # Same commit should generate different sdist
            # depending on tagged version (set $PYSYM_RELEASE_VERSION)
            # e.g.:  $ PYSYM_RELEASE_VERSION=v1.2.3 python setup.py sdist
            # this will ensure source distributions contain the correct version
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__', release_py_path)
