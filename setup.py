#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="visionparse",
    version="1.0.0",
    author="Maharsh Patel",
    author_email="maharsh2017@gmail.com",
    description="Production-ready tool for analyzing screenshots and extracting UI elements using Vision Language Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MaharshPatelX/VisionParse",
    project_urls={
        "Bug Tracker": "https://github.com/MaharshPatelX/VisionParse/issues",
        "Documentation": "https://github.com/MaharshPatelX/VisionParse/blob/main/README.md",
        "Source Code": "https://github.com/MaharshPatelX/VisionParse",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    include_package_data=True,
    package_data={
        "VisionParse": [
            "config.json",
            "*.md",
        ],
    },
    keywords=[
        "computer vision",
        "ui analysis", 
        "screenshot analysis",
        "vlm",
        "vision language model",
        "yolo",
        "object detection",
        "automation",
        "ai",
        "machine learning"
    ],
    license="MIT",
    zip_safe=False,
)