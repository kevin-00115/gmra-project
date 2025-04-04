# SYSTEM IMPORTS
from typing import List, Dict
import torch
from torch.utils import cpp_extension
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import os
import platform
import re
import subprocess
import sys
import sysconfig


# PYTHON PROJECT IMPORTS

"""
    Most code in this file is shamelessly adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
"""

name = "mcas_gmra"
cd = os.path.abspath(os.path.dirname(__file__))
csrc_dir = os.path.join(cd, "csrc")
inc_dir = os.path.join(csrc_dir, "include")
src_dir = os.path.join(csrc_dir, "src")
pybind_dir = os.path.join(csrc_dir, "pybind")
sources = [
           os.path.join(src_dir, "covertree.cc"),
           os.path.join(src_dir, "dyadictree.cc"),
           os.path.join(pybind_dir, "pybind_trees.cc"),
          ]

extension = cpp_extension.CppExtension(
    name=name,
    sources=sources,
    include_dirs=cpp_extension.include_paths() + [inc_dir],
    extra_compile_args=["-std=c++17"],
    language="c++"
)

setup(
    name = name,
    version="0.0.1",
    author="Andrew Wood",
    author_email="aewood@bu.edu",
    description="Adaptation of GMRA (https://mauromaggioni.duckdns.org/code/) to Python in C++ using MCAS, pybind11, CMake, and libtorch",
    long_description="",
    ext_modules=[extension],
    cmdclass=dict(build_ext=cpp_extension.BuildExtension),
)

