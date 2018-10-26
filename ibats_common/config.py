# -*- coding: utf-8 -*-
"""
Created on 2017/6/9
@author: MG
"""
import logging
from logging.config import dictConfig

logger = logging.getLogger()


class ConfigBase:

    DB_SCHEMA_IBATS = 'ibats'
    DB_URL_DIC = {
        DB_SCHEMA_IBATS: 'mysql://mg:****@localhost/' + DB_SCHEMA_IBATS,
    }

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
    dictConfig(logging_config)


# 开发配置（SIMNOW MD + Trade）
config = ConfigBase()


def update_config(config_update: ConfigBase):
    global config
    config = config_update
    logger.info('更新默认配置信息 %s < %s', ConfigBase, config_update)
