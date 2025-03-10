#!/usr/bin/env python3
"""
Setup script for py_primitive package.
"""
from setuptools import setup, find_packages

setup(
    name="py_primitive",
    version="0.1.0",
    description="GPU-accelerated image approximation with geometric primitives",
    author="Primitive Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/primitive",
    packages=find_packages(include=["py_primitive", "py_primitive.*"]),
    package_data={"py_primitive": ["examples/*.py"]},
    entry_points={
        "console_scripts": [
            "py_primitive=py_primitive.main:main",
        ],
    },
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 