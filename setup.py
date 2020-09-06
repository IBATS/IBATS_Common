#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/14 16:07
@File    : setup.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as rm:
    long_description = rm.read()

setup(name='IBATS_Common',
      version='0.18.5',
      description='IBATS（Integration Backtest Analysis Trade System）的公共模块，所有Feeder, Trader均集成这个模块，并使用其提供的一些公共工具',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='MG',
      author_email='mmmaaaggg@163.com',
      url='https://github.com/IBATS/IBATS_Common',
      packages=find_packages(),
      python_requires='>=3.6',
      classifiers=(
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: Unix",
          "Operating System :: POSIX",
          "License :: OSI Approved :: MIT License",
          "Development Status :: 5 - Production/Stable",
          "Environment :: No Input/Output (Daemon)",
          "Intended Audience :: Developers",
          "Natural Language :: Chinese (Simplified)",
          "Topic :: Software Development",
      ),
      install_requires=[
          'numpy>=1.16.2',
          'pandas>=0.24.2',
          'redis>=2.10.6',
          'SQLAlchemy>=1.3.2',
          'mysqlclient>=1.4.6',
          'xlrd>=1.1.0',
          'ibats_utils>=1.2.8',
          'matplotlib>=3.0.3',
          'seaborn>=0.9',
          'ffn>=0.3.4',
          'python-docx>=0.8.10',
          'TA-Lib>=0.4.17',
          'tensorflow>=2.2.0',
      ])
