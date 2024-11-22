from setuptools import setup

setup(
    name='pytorch-cpr',
    version='0.3.1',
    description='Constrained Parameter Regularization for PyTorch',
    url='https://github.com/automl/CPR',
    author='Joerg Franke',
    license='Apache License 2.0',
    packages=['pytorch_cpr'],
    install_requires=['pytorch>=2.0.0'],
)
