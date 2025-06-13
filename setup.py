#!/usr/bin/env python3
"""
Setup script for Practical Task P2 - Advanced NumPy Toolkit

Author: George Dorochov
Email: jordanaftermidnight@gmail.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="practical-task-p2",
    version="1.0.0",
    author="George Dorochov",
    author_email="jordanaftermidnight@gmail.com",
    description="Advanced NumPy project with simulation proof and data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jordanaftermidnight/Practical-Task-P2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "practical-task-p2=examples.complete_demonstration:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="numpy data-analysis simulation image-processing statistics",
    project_urls={
        "Bug Reports": "https://github.com/jordanaftermidnight/Practical-Task-P2/issues",
        "Source": "https://github.com/jordanaftermidnight/Practical-Task-P2",
    },
)