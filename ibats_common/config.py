# -*- coding: utf-8 -*-
"""
Created on 2017/6/9
@author: MG
"""
import logging
from logging.config import dictConfig
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigBase:

    DB_SCHEMA_IBATS = 'ibats'
    DB_URL_DIC = {
        DB_SCHEMA_IBATS: 'mysql://m*:****@localhost/' + DB_SCHEMA_IBATS,
    }

    BACKTEST_UPDATE_OR_INSERT_PER_ACTION = False
    ORM_UPDATE_OR_INSERT_PER_ACTION = True
    UPDATE_STG_RUN_STATUS_DETAIL_PERIOD = 1  # 1 每一个最小行情周期，2 每天

    # evn configuration
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(filename)s.%(funcName)s:%(lineno)d|%(message)s'

    # log settings
    logging_config = dict(
        version=1,
        formatters={
            'simple': {
                'format': LOG_FORMAT}
        },
        handlers={
            'file_handler':
                {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'logger.log',
                    'maxBytes': 1024 * 1024 * 10,
                    'backupCount': 5,
                    'level': 'DEBUG',
                    'formatter': 'simple',
                    'encoding': 'utf8'
                },
            'console_handler':
                {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'simple'
                }
        },

        root={
            'handlers': ['console_handler', 'file_handler'],
            'level': logging.DEBUG,
        }
    )
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARN)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    dictConfig(logging_config)
    update_db_datetime = datetime.now()


# 开发配置（SIMNOW MD + Trade）
config = ConfigBase()


def update_config(config_new: ConfigBase):
    """更新配置信息"""
    global config
    config = config_new
    config.update_db_datetime = datetime.now()
    logger.info('更新默认配置信息 %s < %s', ConfigBase, config_new.__class__)


def update_db_config(db_url_dic: dict):
    """更新数据配置链接"""
    config.DB_URL_DIC.update(db_url_dic)
    logger.debug('更新数据库配置信息  %s keys: %s', ConfigBase, list(db_url_dic.keys()))
    config.update_db_datetime = datetime.now()
