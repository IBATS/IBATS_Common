# -*- coding: utf-8 -*-
"""
Created on 2017/6/9
@author: MG
"""
import logging
from ibats_common.config import ConfigBase as ConBase, update_db_config
from ibats_common.common import ExchangeName

logger = logging.getLogger()


class ConfigBase(ConBase):
    # 交易所名称
    MARKET_NAME = ExchangeName.LocalFile.name

    def __init__(self):
        super().__init__()
        # 设定默认日志格式
        logging.basicConfig(level=logging.DEBUG, format=self.LOG_FORMAT)
        # 设置rest调用日志输出级别
        logging.getLogger('EventAgent').setLevel(logging.INFO)
        logging.getLogger('StgBase').setLevel(logging.INFO)


# 开发配置（SIMNOW MD + Trade）
config = ConfigBase()
update_db_config(config.DB_URL_DIC)


def update_config(config_new: ConfigBase, update_db=True):
    global config
    config = config_new
    logger.info('更新默认配置信息 %s < %s', ConfigBase, config_new.__class__)
    if update_db:
        update_db_config(config.DB_URL_DIC)
