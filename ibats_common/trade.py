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
from collections import OrderedDict
from datetime import datetime
from functools import partial
from ibats_common.config import config
from ibats_common.backend.orm import OrderDetail, engine_ibats, TradeDetail, PosStatusDetail, AccountStatusDetail
from ibats_common.common import RunMode, ExchangeName, BacktestTradeMode, Action, Direction, PositionDateType
from ibats_common.utils.db import with_db_session

logger = logging.getLogger(__package__)


class TraderAgentBase(ABC):
    """
    交易代理（抽象类），回测交易代理，实盘交易代理的父类
    """

    def __init__(self, stg_run_id, exchange_name, **agent_params):
        """
        stg_run_id 作为每一次独立的执行策略过程的唯一标识
        :param stg_run_id:
        :param exchange_name:
        """
        self.stg_run_id = stg_run_id
        self.agent_params = agent_params
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange_name = exchange_name

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


class BacktestTraderAgentBase(TraderAgentBase):
    """
    供调用模拟交易接口使用
    """

    def __init__(self, stg_run_id, **agent_params):
        super().__init__(stg_run_id, **agent_params)
        # 标示 order 成交模式
        self.trade_mode = agent_params.setdefault('trade_mode', BacktestTradeMode.Order_2_Deal)
        # 账户初始资金
        self.init_cash = agent_params['init_cash']
        # 用来标示当前md，一般执行买卖交易是，对时间，价格等信息进行记录
        self.curr_md_period_type = None
        self.curr_md = None
        # 用来保存历史的 order_detail trade_detail pos_status_detail account_info
        self.order_detail_list = []
        self.trade_detail_list = []
        self.pos_status_detail_dic = OrderedDict()
        self.account_info_list = []
        # 持仓信息 初始化持仓状态字典，key为 symbol
        self._pos_status_detail_dic = {}
        self._order_detail_dic = {}
        # 账户信息
        self._account_status_detail = None
        # 关键 key 信息
        self.timestamp_key = None
        self.symbol_key = None
        self.close_key = None
        # 未成交的订单列表
        self.un_finished_order_list = []

    def set_curr_md(self, period_type, md):
        self.curr_md_period_type = period_type
        self.curr_md = md
        if self.trade_mode != BacktestTradeMode.Order_2_Deal:
            # 根据行情判断是否订单成交
            finished_order_list = []
            for order_detail in self.un_finished_order_list:
                # 如果 开多 或 平空 情况下
                if (order_detail.direction == int(Direction.Long) and order_detail.action == int(Action.Open)) or (
                        order_detail.direction == int(Direction.Short) and order_detail.action != int(Action.Open)
                ):
                    # 目前设置只有价格超越订单价格才能成就，= 的情况不能成就
                    if order_detail.order_price < md[self.close_key]:
                        self._record_trade_detail(order_detail)
                        finished_order_list.append(order_detail)
                else:
                    # 目前设置只有价格跌破订单价格才能成就，= 的情况不能成就
                    if order_detail.order_price > md[self.close_key]:
                        self._record_trade_detail(order_detail)
                        finished_order_list.append(order_detail)

            for order_detail in finished_order_list:
                self.un_finished_order_list.remove(order_detail)

    def set_timestamp_key(self, key):
        self.timestamp_key = key

    def set_symbol_key(self, key):
        self.symbol_key = key

    def set_close_key(self, key):
        self.close_key = key

    def connect(self):
        pass

    def check_key(self):
        if self.timestamp_key is None:
            raise ValueError('timestamp_key 尚未设置，请先调用 set_timestamp_key')
        if self.symbol_key is None:
            raise ValueError('symbol_key 尚未设置，请先调用 set_symbol_key')
        if self.close_key is None:
            raise ValueError('close_key 尚未设置，请先调用 set_close_key')

    @property
    def curr_timestamp(self) -> datetime:
        return self.curr_md[self.timestamp_key]

    @property
    def curr_symbol(self) -> str:
        return self.curr_md[self.symbol_key]

    @property
    def curr_close(self) -> float:
        return float(self.curr_md[self.close_key])

    def _record_order_detail(self, symbol, price: float, vol: int, direction: Direction, action: Action):
        order_date = self.curr_timestamp.date()
        order_detail = OrderDetail(stg_run_id=self.stg_run_id,

                                 order_date=order_date,
                                 order_time=self.curr_timestamp.time(),
                                 order_millisec=0,
                                 direction=int(direction),
                                 action=int(action),
                                 symbol=symbol,
                                 order_price=float(price),
                                 order_vol=int(vol)
                                 )
        if config.BACKTEST_UPDATE_OR_INSERT_PER_ACTION:
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(order_detail)
                session.commit()
        self.order_detail_list.append(order_detail)
        self._order_detail_dic.setdefault(symbol, []).append(order_detail)
        # 更新成交信息
        # Order_2_Deal 模式：下单即成交
        if self.trade_mode == BacktestTradeMode.Order_2_Deal:
            self._record_trade_detail(order_detail)
        else:
            self.un_finished_order_list.append(order_detail)

    def _record_trade_detail(self, order_detail: OrderDetail):
        """
        根据订单信息保存成交结果
        :param order_detail:
        :return:
        """
        trade_detail = TradeDetail.create_by_order_detail(order_detail)
        self.trade_detail_list.append(trade_detail)
        # 更新持仓信息
        self._record_pos_status_detail(trade_detail)

    def _record_pos_status_detail(self, trade_detail: TradeDetail):
        """
        根据成交信息保存最新持仓信息
        :param trade_detail:
        :return:
        """
        symbol = trade_detail.symbol
        if symbol in self._pos_status_detail_dic:
            pos_status_detail_last = self._pos_status_detail_dic[symbol]
            pos_status_detail = pos_status_detail_last.update_by_trade_detail(trade_detail)
        else:
            pos_status_detail = PosStatusDetail.create_by_trade_detail(trade_detail)
        # 更新
        trade_date, trade_time, trade_millisec = \
            pos_status_detail.trade_date, pos_status_detail.trade_time, pos_status_detail.trade_millisec
        self.pos_status_detail_dic[(trade_date, trade_time, trade_millisec)] = pos_status_detail
        self._pos_status_detail_dic[symbol] = pos_status_detail
        # self.c_save_acount_info(pos_status_detail)

    def _record_account_status_detail(self) -> AccountStatusDetail:
        stg_run_id, init_cash = self.stg_run_id, self.init_cash
        timestamp_curr = self.curr_timestamp
        trade_date = timestamp_curr.date()
        trade_time = timestamp_curr.time()
        trade_millisec = 0
        # trade_price = float(self.curr_md['close'])
        acc_status_info = AccountStatusDetail(stg_run_id=stg_run_id,
                                              trade_date=trade_date,
                                              trade_time=trade_time,
                                              trade_millisec=trade_millisec,
                                              available_cash=init_cash,
                                              balance_tot=init_cash,
                                              )
        if config.BACKTEST_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(acc_status_info)
                session.commit()
        return acc_status_info

    def _update_by_pos_status_detail(self) -> AccountStatusDetail:
        """根据 持仓列表更新账户信息"""

        pos_status_detail_dic = self._pos_status_detail_dic
        timestamp_curr = self.curr_timestamp
        account_status_detail = self._account_status_detail.create_by_self()
        # 上一次更新日期、时间
        # trade_date_last, trade_time_last, trade_millisec_last = \
        #     account_status_detail.trade_date, account_status_detail.trade_time, account_status_detail.trade_millisec
        # 更新日期、时间
        trade_date = timestamp_curr.date()
        trade_time = timestamp_curr.time()
        trade_millisec = 0

        curr_margin = 0
        close_profit = 0
        position_profit = 0
        floating_pl_chg = 0
        margin_chg = 0
        floating_pl_cum = 0
        for symbol, pos_status_detail in pos_status_detail_dic.items():
            curr_margin += pos_status_detail.margin
            if pos_status_detail.position == 0:
                close_profit += pos_status_detail.floating_pl
            else:
                position_profit += pos_status_detail.floating_pl
            floating_pl_chg += pos_status_detail.floating_pl_chg
            margin_chg += pos_status_detail.margin_chg
            floating_pl_cum += pos_status_detail.floating_pl_cum

        available_cash_chg = floating_pl_chg - margin_chg
        account_status_detail.curr_margin = curr_margin
        # # 对于同一时间，平仓后又开仓的情况，不能将close_profit重置为0
        # if trade_date == trade_date_last and trade_time == trade_time_last and trade_millisec == trade_millisec_last:
        #     account_status_detail.close_profit += close_profit
        # else:
        # 一个单位时段只允许一次，不需要考虑上面的情况
        account_status_detail.close_profit = close_profit

        account_status_detail.position_profit = position_profit
        account_status_detail.available_cash += available_cash_chg
        account_status_detail.floating_pl_cum = floating_pl_cum
        account_status_detail.balance_tot = account_status_detail.available_cash + curr_margin

        account_status_detail.trade_date = trade_date
        account_status_detail.trade_time = trade_time
        account_status_detail.trade_millisec = trade_millisec
        if config.BACKTEST_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(account_status_detail)
                session.commit()
        return account_status_detail

    def _update_pos_status_detail_by_md(self, pos_status_detail_last) -> PosStatusDetail:
        """创建新的对象，根据 trade_detail 更新相关信息"""
        timestamp_curr = self.curr_timestamp
        trade_date = timestamp_curr.date()
        trade_time = timestamp_curr.time()
        trade_millisec = 0
        trade_price = self.curr_close
        # symbol = md['symbol']

        pos_status_detail = pos_status_detail_last.create_by_self()
        pos_status_detail.cur_price = trade_price
        pos_status_detail.trade_date = trade_date
        pos_status_detail.trade_time = trade_time
        pos_status_detail.trade_millisec = trade_millisec

        # 计算 floating_pl margin
        # instrument_info = config.instrument_info_dic[symbol]
        # multiple = instrument_info['VolumeMultiple']
        # margin_ratio = instrument_info['LongMarginRatio']
        multiple, margin_ratio = 1, 1
        position = pos_status_detail.position
        cur_price = pos_status_detail.cur_price
        avg_price = pos_status_detail.avg_price
        pos_status_detail.margin = position * cur_price * multiple * margin_ratio
        pos_status_detail.margin_chg = pos_status_detail.margin - pos_status_detail_last.margin
        if pos_status_detail.direction == Direction.Long:
            pos_status_detail.floating_pl = (cur_price - avg_price) * position * multiple
        else:
            pos_status_detail.floating_pl = (avg_price - cur_price) * position * multiple
        pos_status_detail.floating_pl_chg = pos_status_detail.floating_pl - pos_status_detail_last.floating_pl
        pos_status_detail.floating_pl_cum += pos_status_detail.floating_pl_chg

        if config.BACKTEST_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(pos_status_detail)
                session.commit()
        return pos_status_detail

    def update_account_info(self):
        """
        更新 持仓盈亏数据 汇总统计当前周期账户盈利情况
        :return:
        """
        if self.curr_md is None:
            return
        if self._account_status_detail is None:
            # self._account_status_detail = AccountStatusInfo.create(self.stg_run_id, self.init_cash, self.curr_md)
            self._account_status_detail = self._record_account_status_detail()
            self.account_info_list.append(self._account_status_detail)

        symbol = self.curr_symbol
        if symbol in self._pos_status_detail_dic:
            pos_status_detail_last = self._pos_status_detail_dic[symbol]
            trade_date = pos_status_detail_last.trade_date
            trade_time = pos_status_detail_last.trade_time
            # 如果当前K线以及更新则不需再次更新。如果当前K线以及有交易产生，则 pos_info 将会在 _save_pos_status_detail 函数中被更新，因此无需再次更新
            if trade_date == self.curr_timestamp.date() and trade_time == self.curr_timestamp.time():
                return
            # 说明上一根K线位置已经平仓，下一根K先位置将记录清除
            if pos_status_detail_last.position == 0:
                del self._pos_status_detail_dic[symbol]
            # 根据 md 数据更新 仓位信息
            # pos_status_detail = pos_status_detail_last.update_by_md(self.curr_md)
            pos_status_detail = self._update_pos_status_detail_by_md(pos_status_detail_last)
            self._pos_status_detail_dic[symbol] = pos_status_detail

        # 统计账户信息，更新账户信息
        # account_status_detail = self._account_status_detail.update_by_pos_status_detail(
        #     self._pos_status_detail_dic, self.curr_md)
        account_status_detail = self._update_by_pos_status_detail()
        self._account_status_detail = account_status_detail
        self.account_info_list.append(self._account_status_detail)

    def open_long(self, symbol, price, vol):
        self._record_order_detail(symbol, price, vol, Direction.Long, Action.Open)

    def close_long(self, symbol, price, vol):
        self._record_order_detail(symbol, price, vol, Direction.Long, Action.Close)

    def open_short(self, symbol, price, vol):
        self._record_order_detail(symbol, price, vol, Direction.Short, Action.Open)

    def close_short(self, symbol, price, vol):
        self._record_order_detail(symbol, price, vol, Direction.Short, Action.Close)

    def get_position(self, symbol, **kwargs) -> dict:
        if symbol in self._pos_status_detail_dic:
            pos_status_detail = self._pos_status_detail_dic[symbol]
            position_date_pos_info_dic = {PositionDateType.History: pos_status_detail}
        else:
            position_date_pos_info_dic = None
        return position_date_pos_info_dic

    @property
    def datetime_last_update_position(self) -> datetime:
        return datetime.now()

    @property
    def datetime_last_rtn_trade_dic(self) -> dict:
        raise NotImplementedError()

    @property
    def datetime_last_update_position_dic(self) -> dict:
        raise NotImplementedError()

    @property
    def datetime_last_send_order_dic(self) -> dict:
        raise NotImplementedError()

    def release(self):
        try:
            with with_db_session(engine_ibats) as session:
                session.add_all(self.order_detail_list)
                session.add_all(self.trade_detail_list)
                session.add_all(self.pos_status_detail_dic.values())
                session.add_all(self.account_info_list)
                session.commit()
        except:
            self.logger.exception("release exception")

    def get_order(self, symbol) -> (OrderDetail, None):
        if symbol in self._order_detail_dic:
            return self._order_detail_dic[symbol]
        else:
            return None

    def get_balance(self):
        position_date_pos_info_dic = {key: {PositionDateType.History: pos_status_detail}
                                      for key, pos_status_detail in self._pos_status_detail_dic.items()}
        return position_date_pos_info_dic


trader_agent_class_dic = {
    RunMode.Backtest: {ExchangeName.Default: TraderAgentBase},
    RunMode.Realtime: {ExchangeName.Default: TraderAgentBase}
}


def trader_agent_factory(run_mode: RunMode, stg_run_id, exchange_name: ExchangeName,
                         **trade_agent_params) -> TraderAgentBase:
    """工厂类用来生成相应 TraderAgentBase 实例"""
    trader_agent_class = trader_agent_class_dic[run_mode][exchange_name]
    trader_agent_obj = trader_agent_class(stg_run_id, exchange_name=exchange_name, **trade_agent_params)
    return trader_agent_obj


def register_trader_agent(agent: TraderAgentBase, run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default,
                          is_default=True) -> TraderAgentBase:
    """注册 TraderAgent"""
    trader_agent_class_dic[run_mode][exchange_name] = agent
    if is_default:
        trader_agent_class_dic[run_mode][ExchangeName.Default] = agent
    logger.info('注册 %s trade agent[%s] = %s', run_mode, exchange_name, agent)
    return agent


def trader_agent(run_mode: RunMode, exchange_name: ExchangeName = ExchangeName.Default, is_default=True):
    """用来注册 TraderAgent 的装饰器"""
    func = partial(register_trader_agent, run_mode=run_mode, exchange_name=exchange_name, is_default=is_default)
    return func
