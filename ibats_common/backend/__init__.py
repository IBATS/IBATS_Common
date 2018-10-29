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
from ibats_common.config import config
import logging

logger = logging.getLogger()
engines = {}
engine_ibats = None


def reload_engine(refresh=False):
    """
    重新加载全部引擎
    :param refresh: True：全部重新加载；False：仅增量加载
    :return:
    """
    global engines, engine_ibats
    if refresh:
        engines = {}
    for key, url in config.DB_URL_DIC.items():
        if not refresh and key in engines:
            continue
        logger.debug('加载 engine %s: %s', key, url)
        engines[key] = create_engine(url)

    engine_ibats = engines[config.DB_SCHEMA_IBATS]


reload_engine()
