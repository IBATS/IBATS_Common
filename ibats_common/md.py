#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/20 15:12
@File    : md.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from datetime import timedelta, datetime
from ibats_common.common import PeriodType, RunMode, ExchangeName
from threading import Thread
import time
import pandas as pd
import logging
from abc import ABC, abstractmethod
from functools import partial
from ibats_utils.mess import str_2_date, active_coroutine

logger = logging.getLogger(__package__)


class MdAgentBase(Thread, ABC):

    def __init__(self, instrument_id_list, md_period: PeriodType, exchange_name, agent_name=None,
                 init_load_md_count=None, init_md_date_from=None, init_md_date_to=None, **kwargs):
        if agent_name is None:
            agent_name = f'{exchange_name}.{md_period}'
        self.exchange_name = exchange_name
        super().__init__(name=agent_name, daemon=True)
        self.md_period = md_period
        self.keep_running = None
        self.instrument_id_list = instrument_id_list
        self.init_load_md_count = int(init_load_md_count) if init_load_md_count is not None else None
        self.init_md_date_from = str_2_date(init_md_date_from)
        self.init_md_date_to = str_2_date(init_md_date_to)
        self.logger = logging.getLogger(str(self.__class__))
        self.agent_name = agent_name
        self.params = kwargs
        # 关键 key 信息
        self.timestamp_key = kwargs['timestamp_key'] if 'timestamp_key' in kwargs else None
        self.symbol_key = kwargs['symbol_key'] if 'symbol_key' in kwargs else None
        self.close_key = kwargs['close_key'] if 'close_key' in kwargs else None

    def check_key(self):
        """检查 关键 key 信息 是否设置齐全"""
        if self.timestamp_key is None:
            raise ValueError('timestamp_key 尚未设置')
        if self.symbol_key is None:
            raise ValueError('symbol_key 尚未设置')
        if self.close_key is None:
            raise ValueError('close_key 尚未设置')

    @abstractmethod
    def load_history(self, date_from=None, date_to=None, load_md_count=None) -> (pd.DataFrame, dict):
        """
        从mysql中加载历史数据
        实时行情推送时进行合并后供数据分析使用
        :param date_from: None代表沿用类的 init_md_date_from 属性
        :param date_to: None代表沿用类的 init_md_date_from 属性
        :param load_md_count: 0 代表不限制，None代表沿用类的 init_load_md_count 属性，其他数字代表相应的最大加载条数
        :return: md_df 或者
         ret_data {
            'md_df': md_df, 'datetime_key': 'ts_start',
            'date_key': **, 'time_key': **, 'microseconds_key': **
            }
        """

    def load_history_record(self, date_from=None, date_to=None, load_md_count=None):
        """
        加载历史数据，以协程方式逐条推送出去（生成器方法）
        :param date_from: None代表沿用类的 init_md_date_from 属性
        :param date_to: None代表沿用类的 init_md_date_from 属性
        :param load_md_count: 0 代表不限制，None代表沿用类的 init_load_md_count 属性，其他数字代表相应的最大加载条数
        :return: 加载记录数
        """
        his_df_dic = self.load_history(date_from, date_to, load_md_count)
        md_df = his_df_dic['md_df']
        df_len = md_df.shape[0]
        if df_len == 0:
            return df_len

        datetime_key = his_df_dic['datetime_key'] if 'datetime_key' in his_df_dic else None
        date_key = his_df_dic['date_key'] if 'date_key' in his_df_dic else None
        time_key = his_df_dic['time_key'] if 'time_key' in his_df_dic else None
        microseconds_key = his_df_dic['microseconds_key'] if 'microseconds_key' in his_df_dic else None
        # 字段合法性检查
        if datetime_key is None and date_key is None and time_key is None:
            raise KeyError('load_history 方法返回的 key 无效 %s' % his_df_dic.keys())

        num = 0
        for num, md_s in md_df.iterrows():
            if datetime_key is not None:
                datetime_tag = md_s[datetime_key]
            else:
                datetime_tag = datetime.combine(md_s[date_key], md_s[time_key])

            if microseconds_key is not None:
                datetime_tag += timedelta(microseconds=int(md_df[microseconds_key]))
            yield num, datetime_tag, md_s

        return num

    @active_coroutine
    def cor_load_history_record(self, date_from=None, date_to=None, load_md_count=None):
        """
        加载历史数据，以协程方式逐条推送出去（协程方法）
        :param date_from: None代表沿用类的 init_md_date_from 属性
        :param date_to: None代表沿用类的 init_md_date_from 属性
        :param load_md_count: 0 代表不限制，None代表沿用类的 init_load_md_count 属性，其他数字代表相应的最大加载条数
        :return: 加载记录数        """
        data_count = yield from self.load_history_record(date_from, date_to, load_md_count)
        return data_count

    @abstractmethod
    def connect(self):
        """链接redis、初始化历史数据"""

    @abstractmethod
    def release(self):
        """释放channel资源"""

    def subscribe(self, instrument_id_list=None):
        """订阅合约"""
        if instrument_id_list is None:
            return
        self.instrument_id_list = list(set(self.instrument_id_list) | set(instrument_id_list))

    def unsubscribe(self, instrument_id_list):
        """退订合约"""
        if instrument_id_list is None:
            self.instrument_id_list = []
        else:
            self.instrument_id_list = list(set(self.instrument_id_list) - set(instrument_id_list))


md_agent_class_dic = {_: {ExchangeName.Default: MdAgentBase} for _ in RunMode}


def md_agent_factory(run_mode: RunMode, instrument_id_list: list, md_period: PeriodType, agent_name=None,
                     exchange_name: ExchangeName = ExchangeName.Default, **kwargs) -> MdAgentBase:
    """工厂类用来生成相应 MdAgentBase 实例"""
    md_agent_class = md_agent_class_dic[run_mode][exchange_name]
    md_agent_obj = md_agent_class(
        instrument_id_list=instrument_id_list, md_period=md_period, agent_name=agent_name, exchange_name=exchange_name,
        **kwargs)
    return md_agent_obj


def register_md_agent(agent: MdAgentBase, run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default,
                      is_default=True) -> MdAgentBase:
    """注册 MdAgent"""
    md_agent_class_dic[run_mode][exchange_name] = agent
    if is_default:
        md_agent_class_dic[run_mode][ExchangeName.Default] = agent
    logger.info('注册 %s md agent[%s] = %s', run_mode, exchange_name, agent)
    return agent


def md_agent(run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default, is_default=True):
    """用来注册 MdAgent 的装饰器"""
    func = partial(register_md_agent, run_mode=run_mode, exchange_name=exchange_name, is_default=is_default)
    return func


def _test_only():
    instrument_id_list = ['jm1711', 'rb1712', 'pb1801', 'IF1710']
    md_agent_obj = md_agent_factory(RunMode.Realtime, instrument_id_list, md_period=PeriodType.Min1,
                                    init_load_md_count=100)
    md_df = md_agent_obj.load_history()
    print(md_df.shape)
    md_agent_obj.connect()
    md_agent_obj.subscribe(instrument_id_list)
    md_agent_obj.start()
    for n in range(120):
        time.sleep(1)

    md_agent_obj.keep_running = False
    md_agent_obj.join()
    md_agent_obj.release()
    print("all finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(name)s %(filename)s.%(funcName)s:%(lineno)d|%(message)s')
    _test_only()
