#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/20 15:12
@File    : trade.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from abc import abstractmethod, ABC
from datetime import datetime
from functools import partial
from ibats_common.common import RunMode, ExchangeName

logger = logging.getLogger(__package__)


class TraderAgentBase(ABC):
    """
    交易代理（抽象类），回测交易代理，实盘交易代理的父类
    """

    def __init__(self, stg_run_id, run_mode_params: dict):
        """
        stg_run_id 作为每一次独立的执行策略过程的唯一标识
        :param stg_run_id:
        """
        self.stg_run_id = stg_run_id
        self.run_mode_params = run_mode_params
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def connect(self):
        raise NotImplementedError()

    @abstractmethod
    def open_long(self, instrument_id, price, vol):
        raise NotImplementedError()

    @abstractmethod
    def close_long(self, instrument_id, price, vol):
        raise NotImplementedError()

    @abstractmethod
    def open_short(self, instrument_id, price, vol):
        raise NotImplementedError()

    @abstractmethod
    def close_short(self, instrument_id, price, vol):
        raise NotImplementedError()

    @abstractmethod
    def get_position(self, instrument_id) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_order(self, instrument_id):
        raise NotImplementedError()

    @abstractmethod
    def release(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def datetime_last_update_position(self) -> datetime:
        raise NotImplementedError()

    @property
    @abstractmethod
    def datetime_last_rtn_trade_dic(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def datetime_last_update_position_dic(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def datetime_last_send_order_dic(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def get_balance(self) -> dict:
        raise NotImplementedError()


trader_agent_class_dic = {
    RunMode.Backtest: {ExchangeName.Default: TraderAgentBase},
    RunMode.Realtime: {ExchangeName.Default: TraderAgentBase}
}


def register_trader_agent(agent: TraderAgentBase, run_mode: RunMode, exchange_name: ExchangeName=ExchangeName.Default,
                          is_default=True) -> TraderAgentBase:
    """注册 TraderAgent"""
    trader_agent_class_dic[run_mode][exchange_name] = agent
    if is_default:
        trader_agent_class_dic[run_mode][ExchangeName.Default] = agent
    logger.info('注册 %s trade agent[%s] = %s', run_mode, exchange_name, agent.__class__.__name__)
    return agent


def trader_agent(run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default, is_default=True):
    """用来注册 TraderAgent 的装饰器"""
    func = partial(register_trader_agent, run_mode=run_mode, exchange_name=exchange_name, is_default=is_default)
    return func
