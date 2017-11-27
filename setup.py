#!/usr/bin/env python

"""pennies: a pythonic quantitative finance library for pricing and risk"""

from setuptools import setup, find_packages
import sys

if sys.version_info[:2] < (2,7):
    raise RuntimeError("Currently requires Python version >= 2.7.")

MAJOR = 0
MINOR = 2
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
FULLVERSION = VERSION + '.dev1'

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Financial and Insurance Industry
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Topic :: Software Development
Topic :: Office/Business :: Financial
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


setup(
    name='pennies',
    version=VERSION,
    maintainer='Casey Clements',
    maintainer_email='casey.clements@gmail.com',
    description='pennies: pythonic quantitative finance library',
    long_description=open('README.rst').read(),
    url='https://github.com/caseyclements/pennies',
    packages=find_packages(where='.', exclude='pennies'),
    license='BSD',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=['Mac OS-X'],  # "Windows", "Linux", "Solaris", "Unix"
    install_requires=['numpy', 'pandas', 'scipy', 'multipledispatch']
)

