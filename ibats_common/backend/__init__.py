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
from ibats_utils.db import DynamicEngine
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DyEngine(DynamicEngine):

    def __init__(self, db_url_dic):
        DynamicEngine.__init__(self, db_url_dic)
        self.update_datetime = datetime.now()

    def __getitem__(self, item):
        if config.update_db_datetime > self.update_datetime:
            self.reload_engines(refresh=True)

        return DynamicEngine.__getitem__(self, item)

    def __iter__(self):
        if config.update_db_datetime > self.update_datetime:
            self.reload_engines(refresh=True)

        return DynamicEngine.__iter__(self)

    @property
    def engine_ibats(self):
        return self.__getitem__(config.DB_SCHEMA_IBATS)


engines = DyEngine(config.DB_URL_DIC)
