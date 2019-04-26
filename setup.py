import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyDE",
    version="0.0.1",
    author="Jeff Annis",
    author_email="jannis@mail.usf.edu",
    description="Bayesian inference with differential evolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeff324/pyDE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','scipy','matplotlib']
)