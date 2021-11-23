"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers
	
	Copyright (C) 2021 by the authors: Jarolim Robert (University of Graz), <robert.jarolim@uni-graz.at> and
	Leitner Christoph (Graz University of Technology), <christoph.leitner@tugraz.at>.
	
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
