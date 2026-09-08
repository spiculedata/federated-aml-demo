"""Cython build for the federated transaction-monitoring demo.

Compiles every module of the ``fedxgb`` package into a native extension:

    uv run python setup.py build_ext --inplace

``fedxgb/__init__.py`` is deliberately left as source - it carries only a
docstring, and a plain-Python package initialiser keeps imports predictable.
"""

from __future__ import annotations

import glob

from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

# Annotations are PEP 563 strings throughout this codebase, so Cython must not
# reinterpret them as C type declarations. binding keeps introspection working,
# which dataclasses and argparse both rely on.
COMPILER_DIRECTIVES = {
    "language_level": "3",
    "annotation_typing": False,
    "binding": True,
    "embedsignature": True,
}

C_BUILD_DIR = "build/cython"

SOURCES = sorted(p for p in glob.glob("fedxgb/*.py") if not p.endswith("__init__.py"))


class build_py(_build_py):
    """Install the compiled extensions only - leave the .py sources behind.

    Without this setuptools would copy the original sources in alongside the
    ``.so`` files, so the installed package would still contain readable
    Python and compiling it would have bought nothing.
    """

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [entry for entry in modules if entry[1] == "__init__"]

setup(
    cmdclass={"build_py": build_py},
    ext_modules=cythonize(
        SOURCES,
        compiler_directives=COMPILER_DIRECTIVES,
        # Generated C is kept out of the package directory: it embeds the
        # original Python line-by-line as comments, so shipping it alongside
        # the .so would hand the sources straight back.
        build_dir=C_BUILD_DIR,
        quiet=True,
    ),
    zip_safe=False,
)
