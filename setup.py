import setuptools

with open('README.md', 'rb') as f:
    long_description = f.read().decode('utf-8')

setuptools.setup(
    name="nd_metrics",
    version="0.0.2",
    author="Luciano Vilas Boas Espiridião",
    author_email="lucianovilasboas@gmail.com",
    description="Metrics for Author Name Disambiguation (AND) evaluation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucianovilasboas/nd_metrics",
    packages=["nd_metrics"],#setuptools.find_packages(),
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