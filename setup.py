import sys
from setuptools import setup, Extension
import pybind11

# Compiler flags
cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7'] if sys.platform == 'darwin' else ['-std=c++11', '-O3']

sfc_module = Extension(
    'options_solver',
    sources=['solver.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
)

setup(
    name='options_solver',
    version='1.1',  # <--- CHANGED FROM 1.0 TO 1.1 TO FORCE REBUILD
    description='C++ Extension for Options Pricing',
    ext_modules=[sfc_module],
)
