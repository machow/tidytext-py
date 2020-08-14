#!/usr/bin/env python

import re
import ast
from setuptools import setup, find_namespace_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('tidytext.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

with open('README.md') as f:
    README = f.read()

setup(
    name='tidytext-py',
    version=version,
    py_modules=["tidytext"],
    install_requires=['siuba'],
    description="Text processing with pandas DataFrames.",
    author='Michael Chow',
    author_email='mc_al_github@fastmail.com',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/machow/tidytuesday-py'
    )

