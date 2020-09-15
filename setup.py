#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='TreeMazeAnalyses2',
      version='0.2',
      author='Alex Gonzalez',
      author_email='alx.gnz@gmail.com',
      packages=find_packages(exclude='Notebooks'),
      include_package_data=True,
      platforms='any',
      # install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      )
