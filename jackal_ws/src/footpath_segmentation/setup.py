
# DO NOT USE
# python setup.py install

from setuptools import setup, find_packages


setup(
    name='segmentation',
    version='1.0.0',
    packages=find_packages(where='src'),
    install_requires=['segmentation-models-pytorch']
    )
    
