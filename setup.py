import setuptools
import os
import re

with open('README.md', 'rb') as f:
    long_description = f.read().decode('utf-8')

package = 'nd_metrics'

with open(os.path.join(package, '__init__.py'), 'rb') as f:
    init_py = f.read().decode('utf-8')

version = re.search('^__version__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE).group(1)
author  = re.search('^__author__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE).group(1)
email   = re.search('^__email__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE).group(1)    

setuptools.setup(
    name=package,
    version=version,
    author=author,
    author_email=email,
    description="Metrics for Author Name Disambiguation (AND) evaluation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucianovilasboas/nd_metrics",
    packages=[package],#setuptools.find_packages(),
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license = 'MIT',
    keywords = 'name disambiguation metrics',    
    python_requires='>=3.6',
)