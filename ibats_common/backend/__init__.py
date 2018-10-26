#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/20 15:16
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from sqlalchemy import create_engine
from ibats_trader.config import config
engines = {key: create_engine(url) for key, url in config.DB_URL_DIC.items()}
# engine_md = engines[config.DB_SCHEMA_MD]
engine_ibats = engines[config.DB_SCHEMA_IBATS]
