#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/20 15:12
@File    : md.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_common.common import PeriodType, RunMode, ExchangeName
from threading import Thread
import time
import pandas as pd
import logging
from abc import ABC, abstractmethod
from functools import partial

logger = logging.getLogger(__package__)


class MdAgentBase(Thread, ABC):

    def __init__(self, instrument_id_set, md_period: PeriodType, name=None,
                 init_load_md_count=None, init_md_date_from=None, init_md_date_to=None):
        if name is None:
            name = md_period
        super().__init__(name=name, daemon=True)
        self.md_period = md_period
        self.keep_running = None
        self.instrument_id_set = instrument_id_set
        self.init_load_md_count = init_load_md_count
        self.init_md_date_from = init_md_date_from
        self.init_md_date_to = init_md_date_to
        self.logger = logging.getLogger()

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

    @abstractmethod
    def connect(self):
        """链接redis、初始化历史数据"""

    @abstractmethod
    def release(self):
        """释放channel资源"""

    def subscribe(self, instrument_id_set=None):
        """订阅合约"""
        if instrument_id_set is None:
            return
        self.instrument_id_set |= instrument_id_set

    def unsubscribe(self, instrument_id_set):
        """退订合约"""
        if instrument_id_set is None:
            self.instrument_id_set = set()
        else:
            self.instrument_id_set -= instrument_id_set


md_agent_class_dic = {
    RunMode.Backtest: {ExchangeName.Default: MdAgentBase},
    RunMode.Realtime: {ExchangeName.Default: MdAgentBase}
}


def md_agent_factory(run_mode: RunMode, instrument_id_list, md_period: PeriodType, name=None,
                     exchange_name: ExchangeName = ExchangeName.Default, **kwargs) -> MdAgentBase:
    """工厂类用来生成相应 MdAgentBase 实例"""
    md_agent_class = md_agent_class_dic[run_mode][exchange_name]
    md_agent_obj = md_agent_class(instrument_id_list, md_period, name, **kwargs)
    return md_agent_obj


def register_md_agent(agent: MdAgentBase, run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default,
                      is_default=True) -> MdAgentBase:
    """注册 MdAgent"""
    md_agent_class_dic[run_mode][exchange_name] = agent
    if is_default:
        md_agent_class_dic[run_mode][ExchangeName.Default] = agent
    logger.info('注册 %s md agent[%s] = %s', run_mode, exchange_name, agent.__class__.__name__)
    return agent


def md_agent(run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default, is_default=True):
    """用来注册 MdAgent 的装饰器"""
    func = partial(register_md_agent, run_mode=run_mode, exchange_name=exchange_name, is_default=is_default)
    return func


def _test_only():
    instrument_id_list = {'jm1711', 'rb1712', 'pb1801', 'IF1710'}
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
