import sys
from setuptools import setup, Extension
import pybind11

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
    version='1.3', 
    description='C++ Extension for Options Pricing',
    ext_modules=[sfc_module],
)
