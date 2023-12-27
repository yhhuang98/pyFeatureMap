from setuptools import Extension, setup
from distutils import sysconfig
import numpy

incDirs = [sysconfig.get_python_inc(), numpy.get_include()]

setup(
    ext_modules=[Extension("_cmatrices", ["_cmatrices.c", "cmatrices.c"],include_dirs=incDirs, extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]
)
