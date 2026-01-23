import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Ensure pybind11 is installed before build
try:
    import pybind11
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the setup.py works."""

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'options_solver',
        ['solver.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include() + '//detail',
        ],
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    return True

def cpp_flag(compiler):
    """Return the -std=c++11 compiler flag  and OS specific flags"""
    if sys.platform == 'darwin':
        return ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        # Linux flags (Streamlit Cloud uses Linux)
        return ['-std=c++11', '-O3']

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts = opts + cpp_flag(self.compiler)
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='options_solver',
    version='1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
