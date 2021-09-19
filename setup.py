# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='recoprot',
    version='0.1.0',
    description='Package to run ML algorithms for protein recognition',
    long_description=readme,
    author='Alexandre Dubois',
    author_email='alexandre.dubois@colostate.edu',
    url='https://github.com/aldubois/recoprot',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'env')),
    entry_points={
        'console_scripts': [
            'preprocess = recoprot:preprocess_main',
        ],
    }
)
