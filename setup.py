# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')

setup(
    long_description=readme,
    name='summit',
    version='0.4.0',
    description='Tools for optimizing chemical processes',
    python_requires='==3.*,>=3.6.1',
    project_urls={
        'homepage': 'https://pypi.org/project/summit',
        'repository': 'https://github.com/sustainable-processes/summit'
    },
    author='Kobi Felton',
    author_email='kobi.c.f@gmail.com',
    packages=[
        'summit', 'summit.benchmarks', 'summit.benchmarks.experiment_emulator',
        'summit.strategies', 'summit.utils'
    ],
    package_data={
        'summit.benchmarks.experiment_emulator': [
            'data/*.csv', 'data/*.md', 'data/*.xlsx',
            'trained_models/BNN/*.json', 'trained_models/BNN/*.png',
            'trained_models/BNN/*.pt'
        ]
    },
    install_requires=[
        'blitz-bayesian-pytorch==0.2.3', 'fastprogress==0.*,>=0.2.3',
        'gpy==1.*,>=1.9.0', 'gpyopt==1.*,>=1.2.6', 'gryffin',
        'ipywidgets==7.*,>=7.5.1', 'matplotlib==3.*,>=3.2.2', 'numpy==1.18.0',
<<<<<<< HEAD
        'pandas==1.0.3', 'platypus-opt==1.*,>=1.0.0', 'pymoo==0.*,>=0.4.1',
        'sqsnobfit==0.*,>=0.4.3', 'tensorflow==2.*,>=2.2.0',
        'tensorflow-probability==0.*,>=0.10.1', 'torch', 'tqdm==4.*,>=4.46.1'
=======
        'pandas==1.0.3', 'pathlib==1.*,>=1.0.1', 'pymoo==0.*,>=0.4.1',
        'pyrff==1.*,>=1.0.1', 'sqsnobfit==0.*,>=0.4.3',
        'tensorflow==2.*,>=2.2.0', 'tensorflow-probability==0.*,>=0.10.1',
        'torch', 'tqdm==4.*,>=4.46.1'
>>>>>>> fix_tsemo
    ],
    dependency_links=[
        'git+https://github.com/sustainable-processes/gryffin.git#egg=gryffin'
    ],
    extras_require={
        'dev': [
            'black==19.*,>=19.10.0', 'ipdb==0.*,>=0.13.2',
<<<<<<< HEAD
            'jupyterlab==2.*,>=2.1.4', 'pytest==3.*,>=3.0.0',
            'python-dotenv==0.*,>=0.13.0', 'rope==0.*,>=0.17.0'
=======
            'jupyterlab==2.*,>=2.1.4', 'plotly==4.*,>=4.8.2',
            'pytest==3.*,>=3.0.0', 'python-dotenv==0.*,>=0.13.0',
            'rope==0.*,>=0.17.0'
>>>>>>> fix_tsemo
        ],
        'neptune': [
            'hiplot==0.*,>=0.1.12', 'neptune-client==0.*,>=0.4.115',
            'neptune-contrib[viz]==0.*,>=0.22.0'
        ]
    },
)
