from setuptools import Extension, setup
from distutils import sysconfig
import numpy

incDirs = [sysconfig.get_python_inc(), numpy.get_include()]

setup(
    ext_modules=[Extension("_cshape", ["_cshape.c", "cshape.c"], include_dirs=incDirs)]
)
