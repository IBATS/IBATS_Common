#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/20 15:16
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_common.config import config
from ibats_common.utils.db import DynamicEngine
import logging

logger = logging.getLogger()


class DyEngine(DynamicEngine):

    @property
    def engine_ibats(self):
        return self.__getitem__(config.DB_SCHEMA_IBATS)


engines = DyEngine(config.DB_URL_DIC)
