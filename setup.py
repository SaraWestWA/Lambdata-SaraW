# setup.py file
import setuptools
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Lambdata-SaraW", # the name that you will install via pip
    version="0.0.7",
    author="SaraWestWA",
    author_email="SaraEWestDS@gmail.com",
    description="Very first example package",
    long_description=long_description,
    long_description_content_type="text/markdown", # required if using a md file for long desc
    #license="MIT",
    url="https://github.com/SaraWestWA/Lambdata-SaraW",
    #keywords="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
