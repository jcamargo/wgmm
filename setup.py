#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='shared',
      version='0.0.1',
      description='This package will support wrapped gaussian mixture models. At the moment only a single gaussian and gaussian mixture model are supported',
      author='Julek',
      author_email='juliano.camargo@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE',
    )