# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

def get_version():
    version_file = open('VERSION')
    return version_file.read().strip()
VERSION = get_version()


def get_requirements():
    version_file = open('requirements.txt')
    return version_file.readlines()
requirements = get_requirements()

setup(
    name="minesweeper",
    version=VERSION,
    author="Mathïs Fédérico",
    author_email="mathfederico@gmail.com",
    description="A gym-compatible Minesweeper",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/MathisFederico/Minesweeper",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)