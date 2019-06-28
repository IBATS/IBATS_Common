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
from ibats_common.backend.orm import OrderDetail, engine_ibats, TradeDetail, PosStatusDetail, TradeAgentStatusDetail
from ibats_common.common import RunMode, ExchangeName, BacktestTradeMode, Action, Direction, PositionDateType
from ibats_utils.db import with_db_session
from ibats_utils.mess import date_time_2_str, str_2_datetime
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__package__)


class TraderAgentBase(ABC):
    """
    交易代理（抽象类），回测交易代理，实盘交易代理的父类
    """

    def __init__(self, stg_run_id, exchange_name, agent_name, run_mode=RunMode.Realtime, **agent_params):
        """
        stg_run_id 作为每一次独立的执行策略过程的唯一标识
        :param stg_run_id:
        :param exchange_name:
        :param run_mode: TraderAgentBase 的每一个子类基本有固定的 run_mode，比如:
            TraderAgentBase                     run_mode: RunMode.Realtime
            BacktestTraderAgentBase             run_mode: RunMode.Backtest
            FixPercentBacktestTraderAgentBase   run_mode: RunMode.Backtest_FixPercent
        """
        self.stg_run_id = stg_run_id
        self.agent_params = agent_params
        self.logger = logging.getLogger(str(self.__class__))
        self.exchange_name = exchange_name
        self.agent_name = agent_name
        self.run_mode = run_mode

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

    def __init__(self, stg_run_id, calc_mode, run_mode=RunMode.Backtest, **agent_params):
        super().__init__(stg_run_id, run_mode=run_mode, **agent_params)
        # 标示 order 成交模式
        self.trade_mode = agent_params.setdefault('trade_mode', BacktestTradeMode.Order_2_Deal)
        self.calc_mode = calc_mode
        # 账户初始资金
        self.init_cash = agent_params['init_cash']
        # 用来标示当前md，一般执行买卖交易是，对时间，价格等信息进行记录
        self.curr_md_period_type = None
        self.curr_md = None
        # 用来保存历史的 order_detail trade_detail pos_status_detail account_info
        self.order_detail_list = []
        self.trade_detail_list = []
        self.pos_status_detail_dic = OrderedDict()
        self.trade_agent_detail_list = []
        # 持仓信息 初始化持仓状态字典，key为 symbol
        self._pos_status_detail_dic = {}
        self._order_detail_dic = {}
        # 账户信息
        self.trade_agent_status_detail_latest = None
        # 关键 key 信息
        self.timestamp_key = None
        self.symbol_key = None
        self.close_key = None
        # 未成交的订单列表
        self.un_finished_order_list = []

    def set_curr_md(self, period_type, md):
        self.curr_md_period_type = period_type
        self.curr_md = md
        if self.trade_mode == BacktestTradeMode.MD_2_Deal:
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
        elif self.trade_mode == BacktestTradeMode.NextOpen:
            # 下一个跟K线开盘价作为成交价格
            # finished_order_list = []
            pass

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
                                   trade_agent_key=self.agent_name,
                                   order_date=order_date,
                                   order_time=self.curr_timestamp.time(),
                                   order_millisec=0,
                                   direction=int(direction),
                                   action=int(action),
                                   symbol=symbol,
                                   order_price=float(price),
                                   order_vol=int(vol),
                                   calc_mode=self.calc_mode,
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

    def _update_trade_agent_status_detail_by_pos_status_detail(self) -> TradeAgentStatusDetail:
        """根据 持仓列表更新账户信息"""
        pos_status_detail_dic = self._pos_status_detail_dic
        trade_agent_status_detail = self.trade_agent_status_detail_latest.update_by_pos_status_detail(
            pos_status_detail_dic, self.curr_timestamp)
        return trade_agent_status_detail

    def _update_pos_status_detail_by_md(self, pos_status_detail_last: PosStatusDetail, symbol) -> PosStatusDetail:
        """创建新的对象，根据 trade_detail 更新相关信息"""
        timestamp_curr = self.curr_timestamp
        trade_price = self.curr_close
        pos_status_detail = pos_status_detail_last.update_by_md(trade_price=trade_price, timestamp_curr=timestamp_curr)
        self._pos_status_detail_dic[symbol] = pos_status_detail
        trade_date, trade_time, trade_millisec = \
            pos_status_detail.trade_date, pos_status_detail.trade_time, pos_status_detail.trade_millisec
        self.pos_status_detail_dic[(trade_date, trade_time, trade_millisec)] = pos_status_detail
        return pos_status_detail

    def update_trade_agent_status_detail(self):
        """
        更新 持仓盈亏数据 汇总统计当前周期账户盈利情况
        :return:
        """
        if self.curr_md is None:
            return
        if self.trade_agent_status_detail_latest is None:
            stg_run_id, init_cash = self.stg_run_id, self.init_cash
            timestamp_curr = self.curr_timestamp
            # 首次创建 TradeAgentStatusDetail 需要创建当期交易日 - 1 的 TradeAgentStatusDetail 记录
            trade_agent_status_detail = TradeAgentStatusDetail.create_t_1(
                stg_run_id, trade_agent_key=self.agent_name, init_cash=init_cash, timestamp_curr=timestamp_curr,
                calc_mode=self.calc_mode, run_mode=self.run_mode)
            self.trade_agent_status_detail_latest = trade_agent_status_detail
            self.trade_agent_detail_list.append(self.trade_agent_status_detail_latest)
            # 根据上一交易日 TradeAgentStatusDetail 记录更新当期交易日记录
            trade_agent_status_detail = self._update_trade_agent_status_detail_by_pos_status_detail()
            self.trade_agent_status_detail_latest = trade_agent_status_detail
            self.trade_agent_detail_list.append(self.trade_agent_status_detail_latest)

        symbol = self.curr_symbol
        if symbol in self._pos_status_detail_dic:
            pos_status_detail_last = self._pos_status_detail_dic[symbol]
            # 2019-04-18 每一次行情变化均进行 self._update_pos_status_detail_by_md(pos_status_detail_last) 更新
            # 如果当前K线已经更新则不需再次更新。如果当前K线已经有交易产生，则 pos_info 将会在 _save_pos_status_detail 函数中被更新，因此无需再次更新
            # if trade_date == self.curr_timestamp.date() and trade_time == self.curr_timestamp.time():
            #     return
            # 2019-04-18 已清仓的状态不再清除，而是继续进行后续计算
            # 说明上一根K线位置已经平仓，下一根K先位置将记录清除
            # if pos_status_detail_last.position == 0:
            #     del self._pos_status_detail_dic[symbol]
            # 根据 md 数据更新 仓位信息
            # pos_status_detail = pos_status_detail_last.update_by_md(self.curr_md)
            pos_status_detail = self._update_pos_status_detail_by_md(pos_status_detail_last, symbol)

        # 统计账户信息，更新账户信息
        # trade_agent_status_detail = self.trade_agent_status_detail_latest.update_by_pos_status_detail(
        #     self._pos_status_detail_dic, self.curr_md)
        trade_agent_status_detail = self._update_trade_agent_status_detail_by_pos_status_detail()
        self.trade_agent_status_detail_latest = trade_agent_status_detail
        self.trade_agent_detail_list.append(self.trade_agent_status_detail_latest)
        return self.trade_agent_status_detail_latest

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
        with with_db_session(engine_ibats) as session:
            try:
                session.add_all(self.order_detail_list)
                self.logger.debug("%d 条 order_detail 被保存", len(self.order_detail_list))
                session.commit()
            except SQLAlchemyError:
                logger.exception("%d 条 order_detail 被保存时发生异常", len(self.order_detail_list))
                session.rollback()

            try:
                session.add_all(self.trade_detail_list)
                self.logger.debug("%d 条 trade_detail 被保存", len(self.trade_detail_list))
                session.commit()
            except SQLAlchemyError:
                logger.exception("%d 条 trade_detail 被保存时发生异常", len(self.order_detail_list))
                session.rollback()

            try:
                session.add_all(self.pos_status_detail_dic.values())
                self.logger.debug("%d 条 pos_status_detail 被保存", len(self.pos_status_detail_dic))
                session.commit()
            except SQLAlchemyError:
                logger.exception("%d 条 pos_status_detail 被保存时发生异常", len(self.order_detail_list))
                session.rollback()

            try:
                session.add_all(self.trade_agent_detail_list)
                self.logger.debug("%d 条 trade_agent_detail 被保存", len(self.trade_agent_detail_list))
                session.commit()
            except SQLAlchemyError:
                logger.exception("%d 条 trade_agent_detail 被保存时发生异常", len(self.order_detail_list))
                session.rollback()

    def get_order(self, symbol) -> (OrderDetail, None):
        if symbol in self._order_detail_dic:
            return self._order_detail_dic[symbol]
        else:
            return None

    def get_balance(self):
        position_date_pos_info_dic = {key: {PositionDateType.History: pos_status_detail}
                                      for key, pos_status_detail in self._pos_status_detail_dic.items()}
        return position_date_pos_info_dic


class FixPercentBacktestTraderAgentBase(BacktestTraderAgentBase):
    """
    供调用等比例固定仓位模拟交易接口使用，position取值范围[0-100]，超过100代表加杠杆
    """

    def __init__(self, stg_run_id, calc_mode, run_mode=RunMode.Backtest_FixPercent, **agent_params):
        # 100 不代表 100￥，仅代表 100% 仓位，初始仓位全部为现金
        agent_params['init_cash'] = 1.0
        super().__init__(stg_run_id, calc_mode, run_mode=run_mode, **agent_params)


trader_agent_class_dic = {
    RunMode.Realtime: {ExchangeName.Default: TraderAgentBase},
    RunMode.Backtest: {ExchangeName.Default: BacktestTraderAgentBase},
    RunMode.Backtest_FixPercent: {ExchangeName.Default: FixPercentBacktestTraderAgentBase},
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
