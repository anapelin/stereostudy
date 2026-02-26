"""Setup script for STEREOSTUDYIPSL codebase."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stereostudyipsl",
    version="1.0.0",
    author="Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY",
    author_email="gabriel.jarry@eurocontrol.int, philippe.very@eurocontrol.int",
    description="All-sky camera image processing and stereo cloud height estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eurocontrol/stereostudyipsl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stereo-batch-process=processing.batch_reproject_dataset:main",
            "stereo-visualize=visualization.visualize_azimuth_zenith:main",
        ],
    },
)
