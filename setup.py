import setuptools
from setuptools import setup, find_packages

setup(
    name='deepMTJ',
    version='1.0',
    packages=setuptools.find_packages(),
    url='https://github.com/luuleitner/deepMTJ',
    license='GPL-3.0',
    author='Robert Jarolim, Christoph Leitner',
    description='Automatic tracking of the muscle tendon junction using deep learning',
    install_requires = ['scikit-image', 'scikit-learn', 'tqdm', 'numpy', 'matplotlib', 'keras', 'pandas', 'tensorflow'],
)
