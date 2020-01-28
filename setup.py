#!/usr/bin/env python3
import subprocess
import atexit
from setuptools import setup
from setuptools.command.install import install

def install_punkt():
    """
    Some additional data needs to be installed for the Natural Language
    Tool Kit. This needs to be done after the dependencies have been
    installed. Unfortunately we need to run this in a sub-shell because
    install will fail when it can't "import nltk" during the compilation
    of this script.

    """
    subprocess.check_call(
        """python -c "import nltk; nltk.download('punkt')" """, shell=True)

class CustomInstallCommand(install):
    """Customized setuptools install command - grabs some NLTK data."""

    def __init__(self, *args, **kwargs):
        super(CustomInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(install_punkt)

#    def run(self):
#        install.run(self)
#        atexit.register(install_punkt())

setup(
    cmdclass={'install': CustomInstallCommand},
    name='ArtiClass',
    version='1.0',
    description='Classification of BBC articles.',
    author='Craig MacLachlan',
    author_email='cs.maclachlan@gmail.com',
    packages=['articlass'],
    python_requires='>=3.7',
    install_requires=['tensorflow==1.15.2',
                      'numpy',
                      'scipy',
                      'pandas',
                      'requests',
                      'nltk',
                      'bs4',
                      'matplotlib',
                      'sklearn',
                      'nose2'],
    )