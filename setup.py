# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tf229',
    version='0.0.1',
    description='An implementation of Stanford CS 229 in Tensorflow',
    long_description=readme,
    author='K. N. Tucker',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
