# Copyright (c) InverSQL Authors - All Rights Reserved

import pathlib
import typing
from collections import abc as cabc


import setuptools
from Cython import Build as cb


def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs["ext_modules"] = cb.cythonize(
        "src/inversql/**/*.py",
        compiler_directives={
            "language_level": 3,
        },
    )
