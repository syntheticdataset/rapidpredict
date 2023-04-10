from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.0.2'
DESCRIPTION = 'Generating accurate and safe synthetic datasets for tabular, classification, and time-series labeling tasks'
LONG_DESCRIPTION = 'This repository contains a Python-based framework for generating accurate and safe synthetic datasets for tabular, \
classification, and time-series labeling tasks. It is designed to help researchers, data scientists, and machine learning engineers \
create high-quality, realistic datasets for training and evaluating their models while ensuring privacy and compliance with data protection regulations..'

# Setting up
setup(
    name="rapidpredict",
    version=VERSION,
    author="Synthetic Dataset AI Team",
    author_email="<admin@syntheticdataset.ai>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['python', 'pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'tqdm'],
    keywords=['python', 'pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn'],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: Free To Use But Restricted",

    ] ,
)
