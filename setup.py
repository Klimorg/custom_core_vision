#!/usr/bin/env python
# setup.py
# Setup installation for the application

from pathlib import Path

# import setuptools
from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt"), "r") as req:
    required_packages = [ln.strip() for ln in req.readlines()]

# Load packages from requirements-dev.txt
with open(Path(BASE_DIR, "requirements-dev.txt"), "r") as req_dev:
    dev_packages = [ln.strip() for ln in req_dev.readlines()]

# Load packages from requirements-doc.txt
with open(Path(BASE_DIR, "requirements-doc.txt"), "r") as req_doc:
    doc_packages = [ln.strip() for ln in req_doc.readlines()]


setup(
    name="core_vision",
    version="0.1",
    license="",
    description="installation du projet",
    author="MKL",
    author_email="klimczak.mathieu@pm.me",
    url="",
    package_dir={"core_vision": "core_vision"},
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[required_packages],
    extras_require={
        "dev": [dev_packages] + [doc_packages],
        "docs": [doc_packages],
    },
)
