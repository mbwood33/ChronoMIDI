# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        "kaleidoscope_computations.pyx",
        compiler_directives={'language_level' : "3"} # Ensure Python 3 syntax
    ),
    include_dirs=[numpy.get_include()]
)
