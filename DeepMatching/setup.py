"""
Setup script for stereo_matchers package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stereo_matchers",
    version="0.1.0",
    author="STEREOSTUDYIPSL",
    description="Unified interface for multiple stereo feature matching models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "kornia>=0.7.0",
    ],
    extras_require={
        "roma": ["romatch"],
        "all": ["romatch", "kornia-moons"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "run-matching=run_matching:main",
        ],
    },
)
