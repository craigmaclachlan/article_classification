#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='ArtiClass',
    version='1.0',
    description='Classification of BBC articles.',
    author='Craig MacLachlan',
    author_email='cs.maclachlan@gmail.com',
    packages=['articlass'],
    python_requires='>=3.7',
    install_requires=['tensorflow<2.0',
                      'numpy',
                      'scipy',
                      'pandas',
                      'requests',
                      'nltk',
                      'bs4',
                      'matplotlib',
                      'sklearn'],
     )