# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="transformer_proteomics", 
    version="0.0.1.dev0",
    description="A package for transformer-based proteomics analysis",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="YOUR NAME",
    author_email="YOUR.NAME@email.com",
    url="https://github.com/Jan10e/transformer_proteomics",
    license=license,
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "torch",
        "optuna",
    ],
    zip_safe=False,
    include_package_data=True,
)