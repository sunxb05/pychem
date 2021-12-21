#!/usr/bin/env python
import os

from setuptools import find_packages, setup

setup(
    name="autode",
    packages=find_packages(),
    install_requires=[
    ],
    extras_require={"dev": ["pytest", "black"]},
)
