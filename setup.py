# setup.py
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# Define the Cython extension module
# 'oscilloscope_computations' is the name of the module that will be imported in Python
# 'oscilloscope_computations.pyx' is the Cython file
extensions = [
    Extension(
        "oscilloscope_computations",
        ["oscilloscope_computations.pyx"],
        extra_compile_args=["-O3", "-ffast-math"]
    )
]

setup(ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}, build_dir="build", annotate=False),
    include_dirs=[numpy.get_include()]  # Essential for NumPy support in Cython
)
