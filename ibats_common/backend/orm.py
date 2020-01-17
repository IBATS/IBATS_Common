# -*- coding: utf-8 -*-
"""
Created on 2017/10/9
@author: MG
"""
import logging
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ibats_utils.db import with_db_session
from ibats_utils.mess import str_2_date, pd_timedelta_2_timedelta, datetime_2_str, date_time_2_str, try_2_datetime
from sqlalchemy import Column, Integer, String, DateTime, SmallInteger, Date, Time
from sqlalchemy.dialects.mysql import DOUBLE, TINYINT
from sqlalchemy.ext.declarative import declarative_base

from ibats_common.backend import engines
from ibats_common.common import Action, Direction, CalcMode, ExchangeName, RunMode
from ibats_common.common import PositionDateType
from ibats_common.config import config

logger = logging.getLogger(__name__)
engine_ibats = engines.engine_ibats
BaseModel = declarative_base()
_key_idx = defaultdict(lambda: defaultdict(int))
MAX_RATE = 9.99  # 相当于 999%
_HEART_BEAT_THREAD = None


def idx_generator(key1, key2):
    idx = _key_idx[key1][key2]
    idx += 1
    _key_idx[key1][key2] = idx
    return idx


class StgRunInfo(BaseModel):
    """策略运行信息"""

    __tablename__ = 'stg_run_info'
    stg_run_id = Column(Integer, autoincrement=True, primary_key=True)
    stg_name = Column(String(100))
    stg_module = Column(String(500))
    dt_from = Column(DateTime())
    dt_to = Column(DateTime())
    run_mode = Column(SmallInteger)
    stg_params = Column(String(5000))
    strategy_handler_param = Column(String(1000))
    md_agent_params_list = Column(String(5000))
    trade_agent_params_list = Column(String(5000))


class OrderDetail(BaseModel):
    """订单信息"""

    __tablename__ = 'order_detail'
    stg_run_id = Column(Integer, primary_key=True)
    order_idx = Column(Integer, primary_key=True)
    trade_agent_key = Column(String(40))
    order_dt = Column(DateTime)
    order_date = Column(Date)  # 对应行情数据中 ActionDate
    order_time = Column(Time)  # 对应行情数据中 ActionTime
    order_millisec = Column(Integer, default=0, server_default='0')  # 对应行情数据中 ActionMillisec
    direction = Column(TINYINT)  # -1：空；1：多
    action = Column(Integer)  # 0：开仓：1-6：平仓，具体参见 ibats_common.common.Action 对象
    symbol = Column(String(30))
    order_price = Column(DOUBLE)
    order_vol = Column(DOUBLE)  # 订单量
    calc_mode = Column(TINYINT)  # 计算模式：0 普通模式，1 保证金模式

    def __init__(self, stg_run_id=None, trade_agent_key=None, order_dt=None, order_date=None, order_time=None,
                 order_millisec=0, direction=None, action=None, symbol=None, order_price=0, order_vol=0,
                 calc_mode: (CalcMode, int) = 0):
        self.stg_run_id = stg_run_id
        self.order_idx = None if stg_run_id is None else idx_generator(stg_run_id, OrderDetail)
        self.trade_agent_key = trade_agent_key.name if isinstance(trade_agent_key, ExchangeName) else trade_agent_key
        self.order_dt = date_time_2_str(order_date, order_time) if order_dt is None else order_dt
        self.order_date = order_date
        self.order_time = order_time
        self.order_millisec = order_millisec
        self.direction = direction.value if isinstance(direction, Direction) else direction
        self.action = action.value if isinstance(action, Action) else action
        self.symbol = symbol
        self.order_price = order_price
        self.order_vol = order_vol
        self.calc_mode = calc_mode.value if isinstance(calc_mode, CalcMode) else calc_mode

    def __repr__(self):
        return f"<OrderDetail(stg_run_id='{self.stg_run_id}' idx='{self.order_idx}', " \
               f"trade_agent_key={self.trade_agent_key}, direction='{Direction(self.direction)}', " \
               f"action='{Action(self.action)}', symbol='{self.symbol}', order_price='{self.order_price}', " \
               f"order_vol='{self.order_vol}')>"

    @staticmethod
    def remove(stg_run_id: int):
        """
        仅作为调试工具使用，删除指定 stg_run_id 相关的 order_detail
        :param stg_run_id: 
        :return: 
        """
        with with_db_session(engine_ibats) as session:
            # session.execute('DELETE FROM order_detail WHERE stg_run_id=:stg_run_id',
            #                 {'stg_run_id': stg_run_id})
            session.query(OrderDetail).filter(OrderDetail.stg_run_id == stg_run_id).delete()
            session.commit()


class TradeDetail(BaseModel):
    """记录成交信息"""
    __tablename__ = 'trade_detail'
    stg_run_id = Column(Integer, primary_key=True)
    trade_idx = Column(Integer, primary_key=True)  # , comment="成交id"
    trade_agent_key = Column(String(40))
    order_idx = Column(Integer)  # , comment="对应订单id"
    order_price = Column(DOUBLE)  # , comment="原订单价格"
    order_vol = Column(DOUBLE)  # 订单量 , comment="原订单数量"
    trade_dt = Column(DateTime)  # 对应行情数据中 ActionDate
    trade_date = Column(Date)  # 对应行情数据中 ActionDate
    trade_time = Column(Time)  # 对应行情数据中 ActionTime
    trade_millisec = Column(Integer)  # 对应行情数据中 ActionMillisec
    direction = Column(TINYINT)  # -1：空；1：多
    action = Column(Integer)  # 0：开仓：1-6：平仓，具体参见 ibats_common.common.Action 对象
    symbol = Column(String(30))
    trade_price = Column(DOUBLE)  # , comment="成交价格"
    trade_vol = Column(DOUBLE)  # 订单量 , comment="成交数量"
    margin = Column(DOUBLE, server_default='0')  # 保证金 , comment="占用保证金"
    commission = Column(DOUBLE, server_default='0')  # 佣金、手续费 , comment="佣金、手续费"
    multiple = Column(DOUBLE, server_default='0')  # 合约乘数
    margin_ratio = Column(DOUBLE, server_default='0')  # 保证金比例
    calc_mode = Column(TINYINT)  # 计算模式：0 普通模式，1 保证金模式

    def __init__(self, stg_run_id=None, trade_agent_key=None, order_idx=None, order_price=None, order_vol=None,
                 trade_dt=None, trade_date=None, trade_time=None, trade_millisec=0, direction=None, action=None,
                 symbol=None, trade_price=None, trade_vol=None, margin=None, commission=None, multiple=None,
                 margin_ratio=None, calc_mode: (CalcMode, int) = 0):
        self.stg_run_id = stg_run_id
        self.trade_idx = None if stg_run_id is None else idx_generator(stg_run_id, TradeDetail)
        self.trade_agent_key = trade_agent_key.name if isinstance(trade_agent_key, ExchangeName) else trade_agent_key
        self.order_idx = order_idx
        self.order_price = order_price
        self.order_vol = order_vol
        self.trade_dt = try_2_datetime(date_time_2_str(trade_date, trade_time)) if trade_dt is None else trade_dt
        self.trade_date = trade_date
        self.trade_time = trade_time
        self.trade_millisec = trade_millisec
        self.direction = direction
        self.action = action
        self.symbol = symbol
        self.trade_price = trade_price
        self.trade_vol = trade_vol
        self.margin = margin
        self.commission = commission
        self.multiple = multiple
        self.margin_ratio = margin_ratio
        self.calc_mode = calc_mode.value if isinstance(calc_mode, CalcMode) else calc_mode

    def set_trade_time(self, value):
        if isinstance(value, pd.Timedelta):
            # print(value, 'parse to timedelta')
            self.trade_time = timedelta(seconds=value.seconds)
        else:
            self.trade_time = value

    @staticmethod
    def remove(stg_run_id: int):
        """
        仅作为调试工具使用，删除指定 stg_run_id 相关的 trade_detail
        :param stg_run_id: 
        :return: 
        """
        with with_db_session(engine_ibats) as session:
            # session.execute('DELETE FROM trade_detail WHERE stg_run_id=:stg_run_id',
            #                 {'stg_run_id': stg_run_id})
            # session.query(PosStatusInfo).filter(PosStatusInfo.stg_run_id.in_([3, 4])).delete(False)  # 仅供实例
            session.query(TradeDetail).filter(TradeDetail.stg_run_id == stg_run_id).delete()
            session.commit()

    @staticmethod
    def create_by_order_detail(order_detail: OrderDetail):
        symbol = order_detail.symbol
        order_price, order_vol = order_detail.order_price, order_detail.order_vol
        # stg_run_id = order_detail.stg_run_id

        # TODO: 增加滑点、成交比例、费率、保证金比例等参数，该参数将和在回测参数中进行设置
        # instrument_info = Config.instrument_info_dic[symbol]
        # multiple = instrument_info['VolumeMultiple']
        # margin_ratio = instrument_info['LongMarginRatio']
        multiple, margin_ratio, commission_rate = 1, 1, 0.0005
        margin = order_vol * order_price * multiple * margin_ratio
        commission = order_vol * order_price * multiple * commission_rate
        detail = TradeDetail(stg_run_id=order_detail.stg_run_id,
                             trade_agent_key=order_detail.trade_agent_key,
                             order_idx=order_detail.order_idx,
                             trade_date=order_detail.order_date,
                             trade_time=order_detail.order_time,
                             trade_millisec=order_detail.order_millisec,
                             direction=order_detail.direction,
                             action=order_detail.action,
                             symbol=symbol,
                             order_price=order_price,
                             order_vol=order_vol,
                             trade_price=order_price,
                             trade_vol=order_vol,
                             margin=margin,
                             commission=commission,
                             multiple=multiple,
                             margin_ratio=margin_ratio,
                             calc_mode=order_detail.calc_mode,
                             )
        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()
        return detail


class PosStatusDetail(BaseModel):
    """
    持仓状态数据
    当持仓状态从有仓位到清仓时（position>0 --> position==0），计算清仓前的浮动收益，并设置到 floating_pl 字段最为当前状态的浮动收益
    在调用 create_by_self 时，则需要处理一下，当 position==0 时，floating_pl 直接设置为 0，避免引起后续计算上的混淆
    2018-11-02 当仓位多空切换时（1根K线内多头反手转空头），则浮动收益继续保持之前数字
    """
    __tablename__ = 'pos_status_detail'
    stg_run_id = Column(Integer, primary_key=True)  # 对应回测了策略 StgRunID 此数据与 AccSumID 对应数据相同
    pos_status_detail_idx = Column(Integer, primary_key=True)
    trade_agent_key = Column(String(40))
    trade_idx = Column(Integer)  # , comment="最新的成交id"
    trade_dt = Column(DateTime)  # 每个订单变化生成一条记录 此数据与 AccSumID 对应数据相同
    trade_date = Column(Date)  # 对应行情数据中 ActionDate
    trade_time = Column(Time)  # 对应行情数据中 ActionTime
    trade_millisec = Column(SmallInteger)  # 对应行情数据中 ActionMillisec
    direction = Column(TINYINT)
    symbol = Column(String(30))
    position = Column(DOUBLE, default=0.0)
    position_chg = Column(DOUBLE, default=0.0)
    position_value = Column(DOUBLE, default=0.0)  # 持仓投资品种的总市值 position * trade_price * multiple
    avg_price = Column(DOUBLE, default=0.0)  # 所持投资品种上一交易日所有交易的加权平均价
    cur_price = Column(DOUBLE, default=0.0)
    # 持仓收益，对于普通账户：
    # (trade_price - avg_price_last) * position * int(pos_status_detail.direction) - commission
    # 对于保证金交易：市值增量 - 保证金占比增量
    # (市场价 - 成本价市值) * 仓位 * 方向 -
    floating_pl = Column(DOUBLE, default=0.0)
    # floating_pl_rate 与保证金比例 以及 乘数 变化没有关系
    # (trade_price - avg_price) / avg_price * multiple * int(pos_status_detail.direction)
    floating_pl_rate = Column(DOUBLE, default=0.0)
    floating_pl_chg = Column(DOUBLE, default=0.0)
    floating_pl_cum = Column(DOUBLE, default=0.0)
    # 记录当前状态与前一状态之间净现金流变化情况
    # 例如：
    # 加仓，将导致净现金流为负，减仓则净现金流为正
    # 在保证金交易的情况下，价格波动引起的保证金占用变化，也会使得净现金流产生变化
    cashflow = Column(DOUBLE, default=0.0)
    # 记录每日现金流变化情况
    cashflow_daily = Column(DOUBLE, default=0.0)
    # 累计现金流，整个执行周期内的累计现金净流入量。
    # 该字段将用于与 TradeAgentStatusDetail.cash_init 相加，计算 cash_available 当前可用现金
    cashflow_cum = Column(DOUBLE, default=0.0)
    rr = Column(DOUBLE, default=0.0)  # floating_pl_cum / margin 如果是清仓，则使用前一时刻 margin
    margin = Column(DOUBLE, default=0.0)
    margin_chg = Column(DOUBLE, default=0.0)
    position_date_type = Column(TINYINT, default=0)
    commission = Column(DOUBLE, default=0)  # 当前bar上如果存在交易，则相应的费用记录在此，其他情况均为0
    commission_tot = Column(DOUBLE, default=0)  # 累计费用
    multiple = Column(DOUBLE, server_default='0')  # 合约乘数
    margin_ratio = Column(DOUBLE, server_default='0')  # 保证金比例
    calc_mode = Column(TINYINT)  # 计算模式：0 普通模式，1 保证金模式
    logger = logging.getLogger(f'<Table:{__tablename__}>')

    def __repr__(self):
        return f"<PosStatusDetail(id='{self.pos_status_detail_idx}', trade_agent_key={self.trade_agent_key}, " \
               f"trade_dt='{datetime_2_str(self.trade_dt)}', trade_idx='{self.trade_idx}', symbol='{self.symbol}', " \
               f"direction='{self.direction}', position='{self.position}', avg_price='{self.avg_price}', " \
               f"floating_pl='{self.floating_pl}', floating_pl_chg='{self.floating_pl_chg}', " \
               f"floating_pl_cum='{self.floating_pl_cum}', cashflow='{self.cashflow}', " \
               f"cashflow_daily='{self.cashflow_daily}', cashflow_cum='{self.cashflow_cum}')>"

    def __init__(self, stg_run_id=None, trade_agent_key=None, trade_idx=None, trade_dt=None, trade_date=None,
                 trade_time=None, trade_millisec=None, direction=None, symbol=None, position=None, position_chg=0.0,
                 avg_price=None, cur_price=None, floating_pl=0.0, floating_pl_rate=0.0, floating_pl_chg=0.0,
                 floating_pl_cum=0.0, cashflow=0.0, cashflow_daily=0.0, cashflow_cum=0.0, rr=0.0, margin=0.0,
                 margin_chg=0.0, position_date_type=PositionDateType.Today.value, commission=0.0, commission_tot=0.0,
                 multiple=0, margin_ratio=0.0, calc_mode: (int, CalcMode) = CalcMode.Normal.value):
        self.stg_run_id = stg_run_id
        self.pos_status_detail_idx = None if stg_run_id is None else idx_generator(stg_run_id, PosStatusDetail)
        self.trade_agent_key = trade_agent_key.name if isinstance(trade_agent_key, ExchangeName) else trade_agent_key
        self.trade_idx = trade_idx
        self.trade_dt = date_time_2_str(trade_date, trade_time) if trade_dt is None else trade_dt
        self.trade_date = trade_date
        self.trade_time = trade_time
        self.trade_millisec = trade_millisec
        self.direction = direction
        self.symbol = symbol
        self.position = position
        self.position_chg = position_chg
        self.position_value = position * cur_price
        self.avg_price = avg_price
        self.cur_price = cur_price
        self.floating_pl = floating_pl
        self.floating_pl_rate = floating_pl_rate
        self.floating_pl_chg = floating_pl_chg
        self.floating_pl_cum = floating_pl_cum
        self.cashflow = cashflow
        self.cashflow_daily = cashflow_daily
        self.cashflow_cum = cashflow_cum
        self.rr = rr
        self.margin = margin
        self.margin_chg = margin_chg
        self.position_date_type = position_date_type
        self.commission = commission
        self.commission_tot = commission_tot
        self.multiple = multiple
        self.margin_ratio = margin_ratio
        self.calc_mode = calc_mode.value if isinstance(calc_mode, CalcMode) else calc_mode
        self.last_status = None  # 记录上一个状态实例
        self.last_date_status = None  # 记录上一日最后一个状态实例

    @staticmethod
    def create_by_trade_detail(trade_detail: TradeDetail):
        direction, action, instrument_id = trade_detail.direction, trade_detail.action, trade_detail.symbol
        if action == int(Action.Close):
            raise ValueError('trade_detail.action 不能为 close')
        trade_vol = trade_detail.trade_vol
        trade_price = trade_detail.trade_price
        commission = trade_detail.commission
        tot_value = trade_vol * trade_price
        margin = trade_detail.margin
        # tot_cost = tot_value + commission
        # avg_price = tot_cost / trade_vol
        # 2019-05-19 bug fix on avg_price
        avg_price = (tot_value + commission * int(direction)) / trade_vol
        floating_pl = -commission
        floating_pl_rate = floating_pl / margin
        cashflow = -margin - commission
        detail = PosStatusDetail(stg_run_id=trade_detail.stg_run_id,
                                 trade_agent_key=trade_detail.trade_agent_key,
                                 trade_idx=trade_detail.trade_idx,
                                 trade_dt=trade_detail.trade_dt,
                                 trade_date=trade_detail.trade_date,
                                 trade_time=trade_detail.trade_time,
                                 trade_millisec=trade_detail.trade_millisec,
                                 direction=trade_detail.direction,
                                 symbol=trade_detail.symbol,
                                 position=trade_vol,
                                 position_chg=trade_vol,
                                 avg_price=avg_price,
                                 cur_price=trade_price,
                                 margin=margin,
                                 margin_chg=margin,
                                 floating_pl=floating_pl,
                                 floating_pl_rate=floating_pl_rate,
                                 floating_pl_chg=floating_pl,
                                 floating_pl_cum=floating_pl,
                                 cashflow=cashflow,
                                 cashflow_daily=cashflow,
                                 cashflow_cum=cashflow,
                                 rr=floating_pl_rate,
                                 commission=commission,
                                 commission_tot=commission,
                                 position_date_type=PositionDateType.Today.value,
                                 multiple=trade_detail.multiple,
                                 margin_ratio=trade_detail.margin_ratio,
                                 calc_mode=trade_detail.calc_mode,
                                 )
        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()
        return detail

    def update_by_trade_detail(self, trade_detail: TradeDetail):
        """
        创建新的对象，根据 trade_detail 更新相关信息
        :param trade_detail:
        :return: 
        """
        # 复制前一个持仓状态
        detail = self.create_by_self()
        direction, action, symbol = trade_detail.direction, trade_detail.action, trade_detail.symbol
        trade_price, trade_vol, trade_idx = trade_detail.trade_price, trade_detail.trade_vol, trade_detail.trade_idx
        commission = trade_detail.commission

        # 获取合约信息
        # instrument_info = Config.instrument_info_dic[symbol]
        # multiple = instrument_info['VolumeMultiple']
        # margin_ratio = instrument_info['LongMarginRatio']
        multiple, margin_ratio = 1, 1

        # 计算
        # 仓位 position、position_chg 、
        # 方向 direction、
        # 平均价格 avg_price、
        # 浮动收益 floating_pl floating_pl_rate
        pos_direction_last = self.direction
        position_last = self.position
        avg_price_last = self.avg_price
        margin_last = self.margin

        detail.cur_price = trade_price
        detail.trade_dt = trade_detail.trade_dt
        detail.trade_date = trade_detail.trade_date
        detail.trade_time = trade_detail.trade_time
        detail.trade_millisec = trade_detail.trade_millisec
        if position_last == 0:
            # 如果前一状态仓位为 0 则本次方向与当前订单方向相同
            detail.direction = trade_detail.direction

        direction_int = int(detail.direction)
        if self.calc_mode == CalcMode.Normal.value:
            # 普通模式：非保证金交易模式
            # 普通模式 默认 margin_rate, multiple 均为 1
            if pos_direction_last == direction or position_last == 0:
                if action == Action.Open:
                    # 方向相同：开仓 or 加仓；
                    position_cur = position_last + trade_vol
                    position_value = position_cur * trade_price
                    detail.position_value = position_value
                    detail.position_chg = trade_vol
                    detail.position = position_cur
                    avg_price = (position_last * avg_price_last + trade_price * trade_vol +
                                 commission * direction_int) / position_cur
                    detail.avg_price = avg_price
                    # 计算浮动收益 floating_pl floating_pl_rate
                    detail.floating_pl = (trade_price - avg_price) * position_cur * direction_int
                    detail.floating_pl_rate = (trade_price - avg_price) / avg_price * direction_int
                else:
                    # 方向相反：清仓 or 减仓；
                    detail.position_chg = - trade_vol
                    if trade_vol > position_last:
                        raise ValueError("当前持仓%d，平仓%d，错误" % (position_last, trade_vol))
                    elif trade_vol == position_last:
                        # 清仓
                        position_cur = 0
                        position_value = position_cur * trade_price
                        detail.position_value = position_value
                        detail.avg_price = avg_price = 0
                        detail.position = position_cur
                        # 计算浮动收益 floating_pl floating_pl_rate
                        # 与其他地方计算公式的区别在于 position_curr == 0 因此使用 position_last
                        detail.floating_pl = (trade_price - avg_price_last) * position_last * direction_int - commission
                        detail.floating_pl_rate = (
                                ((trade_price - avg_price_last) * direction_int - commission / position_last)
                                / avg_price_last
                        ) if avg_price_last > 0.001 else MAX_RATE

                    else:
                        # 减仓
                        position_cur = position_last - trade_vol
                        position_value = position_cur * trade_price
                        detail.position_value = position_value
                        avg_price = (position_last * avg_price_last - trade_price * trade_vol + commission
                                     ) / position_cur
                        detail.avg_price = avg_price
                        detail.position = position_cur
                        # 计算浮动收益 floating_pl floating_pl_rate
                        detail.floating_pl = (trade_price - avg_price) * position_cur * direction_int
                        detail.floating_pl_rate = (trade_price - avg_price) / avg_price * direction_int

            else:
                # 方向相反
                raise ValueError("当前仓位：%s %d手，目标操作：%s %d手，请先平仓在开仓" % (
                    "多头" if pos_direction_last == Direction.Long else "空头", position_last,
                    "多头" if direction == Direction.Long else "空头", trade_vol,
                ))

            # 设置其他属性 floating_pl_chg、floating_pl_cum、cur_price、trade_dt、trade_date、trade_time、trade_millisec
            # 2019-05-19 当 position_last == 0 时，代表本次交易为重新开仓
            # 因此，floating_pl_chg = floating_pl，而非与上一状态的差
            detail.floating_pl_chg = (detail.floating_pl - self.floating_pl
                                      ) if position_last != 0 else detail.floating_pl
            detail.floating_pl_cum += detail.floating_pl_chg

            # 计算 position_value、margin、margin_chg
            # cur_price = pos_status_detail.cur_price
            detail.margin = position_cur * trade_price
            # 如果前一状态仓位为 0,  且不是多空切换的情况，则保留上一状态的浮动收益
            if self.position == 0 and self.trade_dt != trade_detail.trade_dt:
                # 新建仓情况下，margin_last 为当前 margin
                # 该变量用于后续计算 rr 使用
                margin_last = detail.margin
                detail.margin_chg = margin_chg = detail.margin
            else:
                margin_last_pos_cur_price = self.position * trade_price
                detail.margin_chg = margin_chg = detail.margin - margin_last_pos_cur_price

            # 计算 cashflow_daily, cashflow_cum, commission, commission_tot, rr, position_date_type
            # 本次现金流
            detail.cashflow = cashflow = - margin_chg - commission
            # 每日现金流
            if self.trade_date != trade_detail.trade_date:
                detail.cashflow_daily = cashflow
            else:
                detail.cashflow_daily += cashflow

            # 累计现金流
            detail.cashflow_cum += cashflow
            detail.commission = commission
            detail.commission_tot += commission
            margin_4_rr = detail.margin if detail.margin > 0 else margin_last
            if margin_4_rr == 0:
                logger.error('%s detail.margin=%f, margin_last=%f',
                             detail.trade_dt, detail.margin, margin_last)
                detail.rr = 0
            else:
                detail.rr = detail.floating_pl_cum / margin_4_rr
            detail.position_date_type = PositionDateType.Today.value

        elif self.calc_mode == CalcMode.Margin.value:
            # 保证金交易模式
            if pos_direction_last == direction or position_last == 0:
                if action == Action.Open:
                    # 方向相同：开仓 or 加仓；
                    position_cur = position_last + trade_vol
                    position_value = position_cur * trade_price * multiple
                    detail.position_value = position_value
                    detail.position_chg = trade_vol
                    detail.position = position_cur
                    avg_price = (position_last * avg_price_last * multiple + trade_price * trade_vol * multiple
                                 + commission * direction_int) / position_cur
                    detail.avg_price = avg_price
                    # 计算 margin、margin_chg
                    margin = position_value * margin_ratio
                    detail.margin = margin
                    margin_chg = margin - margin_last
                    detail.margin_chg = margin_chg
                    # 计算浮动收益 floating_pl floating_pl_rate
                    detail.floating_pl = (trade_price - avg_price) * position_cur * multiple * direction_int
                    detail.floating_pl_rate = (trade_price - avg_price) / avg_price * direction_int
                else:
                    # 方向相反：清仓 or 减仓；
                    detail.position_chg = - trade_vol
                    if trade_vol > position_last:
                        raise ValueError("当前持仓%d，平仓%d，错误" % (position_last, trade_vol))
                    elif trade_vol == position_last:
                        # 清仓
                        position_cur = 0
                        position_value = position_cur * trade_price * multiple
                        detail.position_value = position_value
                        detail.avg_price = avg_price = 0
                        detail.position = position_cur
                        # 计算 margin、margin_chg
                        margin_last_pos_cur_price = position_last * trade_price * multiple * margin_ratio
                        margin = position_value * margin_ratio
                        detail.margin = margin
                        margin_chg = margin - margin_last_pos_cur_price
                        detail.margin_chg = margin_chg
                        # 计算浮动收益 floating_pl floating_pl_rate
                        # 与其他地方计算公式的区别在于 position_curr == 0 因此使用 position_last
                        detail.floating_pl = (trade_price - avg_price_last
                                              ) * position_last * multiple * direction_int - commission
                        detail.floating_pl_rate = (
                                ((trade_price - avg_price_last) * direction_int - commission / position_last)
                                / avg_price_last
                        )if avg_price_last > 0.001 else MAX_RATE

                    else:
                        # 减仓
                        position_cur = position_last - trade_vol
                        position_value = position_cur * trade_price * multiple
                        detail.position_value = position_value
                        avg_price = (position_last * avg_price_last * multiple - trade_price * trade_vol * multiple
                                     + commission) / position_cur
                        detail.avg_price = avg_price
                        detail.position = position_cur
                        # 计算 margin、margin_chg
                        margin = position_value * margin_ratio
                        detail.margin = margin
                        margin_chg = margin - margin_last
                        detail.margin_chg = margin_chg
                        # 计算浮动收益 floating_pl floating_pl_rate
                        detail.floating_pl = (trade_price - avg_price) * position_cur * multiple * direction_int
                        detail.floating_pl_rate = (trade_price - avg_price) / avg_price * direction_int

            else:
                # 方向相反
                raise ValueError("当前仓位：%s %d手，目标操作：%s %d手，请先平仓在开仓" % (
                    "多头" if pos_direction_last == Direction.Long else "空头", position_last,
                    "多头" if direction == Direction.Long else "空头", trade_vol,
                ))

            # 设置其他属性 floating_pl_chg、floating_pl_cum、cur_price、trade_dt、trade_date、trade_time、trade_millisec
            # 2019-05-19 当 position_last == 0 时，代表本次交易为重新开仓
            # 因此，floating_pl_chg = floating_pl，而非与上一状态的差
            detail.floating_pl_chg = (detail.floating_pl - self.floating_pl
                                      ) if position_last != 0 else detail.floating_pl
            detail.floating_pl_cum += detail.floating_pl_chg

            # 计算 cashflow_daily、commission、commission_tot、rr、position_date_type
            # 本次现金流
            detail.cashflow = cashflow = - margin_chg - commission
            # 每日现金流
            if self.trade_date != trade_detail.trade_date:
                detail.cashflow_daily = cashflow
            else:
                detail.cashflow_daily += cashflow

            # 累计现金流
            detail.cashflow_cum += cashflow
            detail.commission += commission
            detail.commission_tot += commission
            detail.rr = detail.floating_pl_cum / (
                detail.margin if detail.margin > 0 else margin_last)
            detail.position_date_type = PositionDateType.Today.value

        else:
            ValueError('calc_mode 不是有效的值 %s', self.calc_mode)

        # self.logger.debug("%s", pos_status_detail)

        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()
        return detail

    def update_by_md(self, trade_price, timestamp_curr: (datetime, pd.Timestamp) = None,
                     md: dict = None,
                     timestamp_key=None, date_key=None, time_key=None, milli_sec_key=None, close_key=None):
        """
        根据行情 更新 pos_status_detail 生成新的对象
        :param trade_price:行情报价
        :param timestamp_curr: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param md: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param timestamp_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param date_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param time_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param milli_sec_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param close_key: price 或 md + close_key 参数 两组参数二选其一
        :return:
        """
        # 更新日期、时间
        if timestamp_curr is None:
            timestamp_curr = md[timestamp_key]

        if timestamp_curr is None:
            trade_date = str_2_date(md[date_key]) - timedelta(days=1)
            trade_time = pd_timedelta_2_timedelta(md[time_key])
            trade_dt = datetime_2_str(date_time_2_str(trade_date, trade_time))
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0
        else:
            trade_date = timestamp_curr.date()
            trade_time = timestamp_curr.time()
            trade_dt = timestamp_curr
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0

        # 设置价格
        if trade_price is None:
            trade_price = md[close_key]

        is_new_day = self.trade_date != trade_date
        detail = self.create_by_self(is_new_day=is_new_day)
        detail.cur_price = trade_price
        detail.trade_date = trade_date
        detail.trade_time = trade_time
        detail.trade_dt = trade_dt
        detail.trade_millisec = trade_millisec

        # 乘数、保证金暂时按1来计算
        # TODO: 增加保证金，乘数参数
        multiple, margin_ratio = 1, 1
        # 没有交易，不产生手续费
        commission = 0
        position_cur, avg_price, direction_int = detail.position, detail.avg_price, int(detail.direction)
        position_value = position_cur * trade_price * multiple
        floating_pl = (trade_price - avg_price) * position_cur * multiple * direction_int
        detail.position_value = position_value
        # 计算 floating_pl
        detail.floating_pl = floating_pl
        # 设置属性 floating_pl_chg、floating_pl_cum
        floating_pl_chg = floating_pl - self.floating_pl
        detail.floating_pl_chg = floating_pl_chg
        detail.floating_pl_cum += floating_pl_chg

        if self.calc_mode == CalcMode.Normal.value:
            # 普通模式：非保证金交易模式
            detail.margin = position_cur * trade_price
            margin_last = self.margin
            detail.margin_chg = margin_chg = detail.margin - margin_last
            # 非保证金模式下，行情变化，不会对现金流产生影响，只会引起浮动收益变化
            detail.cashflow = cashflow = 0
        elif self.calc_mode == CalcMode.Margin.value:
            # 保证金交易模式

            # 计算 position_value、margin、margin_chg
            # cur_price = pos_status_detail.cur_price
            detail.margin = position_cur * trade_price * multiple * margin_ratio
            margin_last = self.margin
            detail.margin_chg = margin_chg = detail.margin - margin_last
            # 计算 cashflow_daily, cashflow_cum, commission, commission_tot, rr, position_date_type
            # 本次现金流
            # 保证金模式下的现金流变化 = 盈利增量 - 保证金增量 - 手续费
            # 盈利增量 = 持仓市值增量 × 方向
            detail.cashflow = cashflow = (position_value - self.position_value
                                          ) * direction_int - margin_chg - commission
            # 每日现金流
            if self.trade_date != trade_date:
                detail.cashflow_daily = cashflow
            else:
                detail.cashflow_daily += cashflow

        else:
            ValueError('calc_mode 不是有效的值 %s', self.calc_mode)

        # 累计现金流
        detail.cashflow_cum += cashflow
        detail.commission = commission
        detail.commission_tot += commission
        # 如果无法计算收益率则暂时记 0
        # 2019-06-18 np.nan 的情况将会在 session.commit() 时导致如下异常，因此，替换为0
        # sqlalchemy.exc.ProgrammingError: (MySQLdb._exceptions.ProgrammingError) nan can not be used with MySQL
        detail.rr = (detail.floating_pl_cum / detail.margin) if detail.margin > 0 else 0

        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()

        return detail

    def create_by_self(self, is_new_day=False):
        """
        创建新的对象
        若当前对象持仓为0（position==0），则 浮动收益部分设置为0
        :param is_new_day:
        :return:
        """
        position = self.position
        if is_new_day:
            cashflow_daily = 0
            position_date_type = PositionDateType.History.value
        else:
            cashflow_daily = self.cashflow_daily
            position_date_type = self.position_date_type

        detail = PosStatusDetail(stg_run_id=self.stg_run_id,
                                 trade_agent_key=self.trade_agent_key,
                                 trade_idx=self.trade_idx,
                                 trade_dt=self.trade_dt,
                                 trade_date=self.trade_date,
                                 trade_time=self.trade_time,
                                 trade_millisec=self.trade_millisec,
                                 direction=self.direction,
                                 symbol=self.symbol,
                                 position=position,
                                 avg_price=self.avg_price,
                                 cur_price=self.cur_price,
                                 floating_pl=self.floating_pl if position > 0 else 0,
                                 floating_pl_rate=0.0,
                                 floating_pl_cum=self.floating_pl_cum,
                                 cashflow=0.0,
                                 cashflow_daily=cashflow_daily,
                                 cashflow_cum=self.cashflow_cum,
                                 margin=self.margin,
                                 margin_chg=0,
                                 position_date_type=position_date_type,
                                 commission=0.0,
                                 commission_tot=self.commission_tot,
                                 multiple=self.multiple,
                                 margin_ratio=self.margin_ratio,
                                 calc_mode=self.calc_mode
                                 )
        detail.last_status = self
        if is_new_day:
            detail.last_date_status = self
        else:
            detail.last_date_status = self.last_date_status

        return detail

    @staticmethod
    def remove(stg_run_id: int):
        """
        仅作为调试工具使用，删除指定 stg_run_id 相关的 pos_status_detail
        :param stg_run_id: 
        :return: 
        """
        with with_db_session(engine_ibats) as session:
            # session.execute('DELETE FROM pos_status_detail WHERE stg_run_id=:stg_run_id',
            #                 {'stg_run_id': stg_run_id})
            session.query(PosStatusDetail).filter(PosStatusDetail.stg_run_id == stg_run_id).delete()
            session.commit()
        # PosStatusInfo.query.filter


class TradeAgentStatusDetail(BaseModel):
    """持仓状态数据"""
    __tablename__ = 'trade_agent_status_detail'
    stg_run_id = Column(Integer, primary_key=True)  # 对应回测了策略 StgRunID 此数据与 AccSumID 对应数据相同
    trade_agent_status_detail_idx = Column(Integer, primary_key=True)
    trade_agent_key = Column(String(40))
    trade_dt = Column(DateTime)
    trade_date = Column(Date)  # 对应行情数据中 ActionDate
    trade_time = Column(Time)  # 对应行情数据中 ActionTime
    trade_millisec = Column(Integer)  # 对应行情数据中 ActionMillisec
    # 可用资金, double
    # detail.cash_init + close_profit - curr_margin - commission + (position_value - curr_margin)
    # 对于没有杠杆的产品 position_value == curr_margin 因此 (position_value - curr_margin) == 0
    cash_available_last_day = Column(DOUBLE, default=0.0)
    cash_available = Column(DOUBLE, default=0.0)
    position_value = Column(DOUBLE, default=0.0)
    curr_margin = Column(DOUBLE, default=0.0)  # 当前保证金总额, double
    close_profit = Column(DOUBLE, default=0.0)
    position_profit = Column(DOUBLE, default=0.0)
    floating_pl_cum = Column(DOUBLE, default=0.0)
    commission_tot = Column(DOUBLE, default=0.0)
    cash_init = Column(DOUBLE, default=0.0)
    cash_and_margin = Column(DOUBLE, default=0.0)
    cashflow_daily = Column(DOUBLE, default=0.0)
    cashflow_cum = Column(DOUBLE, default=0.0)
    rr = Column(DOUBLE, default=0.0)  # Return Rate
    rr_nc = Column(DOUBLE, default=0.0)  # Return Rate Compound
    rr_compound = Column(DOUBLE, default=0.0)  # Return Rate No Commission
    rr_compound_nc = Column(DOUBLE, default=0.0)  # Return Rate Compound No Commission
    calc_mode = Column(TINYINT)  # 计算模式：0 普通模式，1 保证金模式
    logger = logging.getLogger(f'<Table:{__tablename__}>')

    def __init__(self, stg_run_id=None, trade_agent_key=None,
                 trade_dt=None, trade_date=None, trade_time=None, trade_millisec=None, cash_available_last_day=0.0,
                 cash_available=0.0, position_value=0.0, curr_margin=0.0, close_profit=0.0, position_profit=0.0,
                 floating_pl_cum=0.0, cashflow_daily=0.0, cashflow_cum=0.0,
                 commission_tot=0.0, cash_init=0.0, calc_mode: (int, CalcMode) = CalcMode.Normal.value,
                 run_mode: (int, RunMode) = RunMode.Backtest.value):
        self.stg_run_id = stg_run_id
        self.trade_agent_status_detail_idx = None if stg_run_id is None else idx_generator(
            stg_run_id, TradeAgentStatusDetail)
        self.trade_agent_key = trade_agent_key.name if isinstance(trade_agent_key, ExchangeName) else trade_agent_key
        self.trade_dt = date_time_2_str(trade_date, trade_time) if trade_dt is None else trade_dt
        self.trade_date = trade_date
        self.trade_time = trade_time
        self.trade_millisec = trade_millisec
        self.cash_available_last_day = cash_available_last_day
        self.cash_available = cash_available
        self.position_value = position_value
        self.curr_margin = curr_margin
        self.close_profit = close_profit
        self.position_profit = position_profit
        self.floating_pl_cum = floating_pl_cum
        self.commission_tot = commission_tot
        self.cash_init = cash_init
        self.cash_and_margin = cash_available + curr_margin
        self.cashflow_daily = cashflow_daily
        self.cashflow_cum = cashflow_cum
        self.rr = 0  # Return Rate
        self.rr_compound = 0  # Return Rate Compound
        self.rr_nc = 0  # Return Rate No Commission
        self.rr_compound_nc = 0  # Return Rate Compound No Commission
        self.calc_mode = calc_mode.value if isinstance(calc_mode, CalcMode) else calc_mode
        self.run_mode = run_mode.value if isinstance(run_mode, RunMode) else run_mode
        self.pos_status_detail_dic = {}  # 用于记录当期状态对应的 pos_status_detail_dic
        self.last_status = None  # 用于记录上一个状态的实力

    def __repr__(self):
        return f"<TradeAgentStatusDetail(id='{self.trade_agent_status_detail_idx}', " \
               f"trade_agent_key={self.trade_agent_key}, trade_dt='{datetime_2_str(self.trade_dt)}', " \
               f"cash_available='{self.cash_available}', cash_available_last_day='{self.cash_available_last_day}', " \
               f"cashflow_daily='{self.cashflow_daily}', cashflow_cum='{self.cashflow_cum}', " \
               f"cash_and_margin='{self.cashflow_cum}', cash_and_margin='{self.cashflow_cum}', " \
               f"floating_pl_cum='{self.floating_pl_cum}')>"

    @staticmethod
    def create_t_1(stg_run_id, trade_agent_key, init_cash: int, timestamp_curr: (datetime, pd.Timestamp) = None,
                   md: dict = None, timestamp_key=None, date_key=None, time_key=None, milli_sec_key=None,
                   calc_mode: (int, CalcMode) = CalcMode.Normal.value,
                   run_mode: (int, RunMode) = RunMode.Backtest.value):
        """
        根据 md 及 初始化资金 创建对象，默认日期为当前md数据-1天
        :param stg_run_id:
        :param trade_agent_key:
        :param init_cash: 
        :param timestamp_curr: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param md: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param timestamp_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param date_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param time_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param milli_sec_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param calc_mode:
        :param run_mode:
        :return:
        """
        warnings.warn('该函数为范例函数，可以根据实际情况改写', UserWarning)
        if timestamp_curr is not None:
            timestamp_curr -= pd.Timedelta(days=1)
        elif timestamp_key is not None:
            timestamp_curr = md[timestamp_key] - pd.Timedelta(days=1)

        if timestamp_curr is None:
            trade_date = str_2_date(md[date_key]) - timedelta(days=1)
            trade_time = pd_timedelta_2_timedelta(md[time_key])
            trade_dt = datetime_2_str(date_time_2_str(trade_date, trade_time))
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0
        else:
            trade_date = timestamp_curr.date()
            trade_time = timestamp_curr.time()
            trade_dt = timestamp_curr
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0

        # calc_mode = calc_mode.value if isinstance(calc_mode, CalcMode) else calc_mode
        acc_status_detail = TradeAgentStatusDetail(stg_run_id=stg_run_id,
                                                   trade_agent_key=trade_agent_key,
                                                   trade_dt=trade_dt,
                                                   trade_date=trade_date,
                                                   trade_time=trade_time,
                                                   trade_millisec=trade_millisec,
                                                   cash_available_last_day=init_cash,
                                                   cash_available=init_cash,
                                                   position_value=0,
                                                   curr_margin=0,
                                                   close_profit=0,
                                                   position_profit=0,
                                                   cash_init=init_cash,
                                                   calc_mode=calc_mode,
                                                   run_mode=run_mode,
                                                   )
        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(acc_status_detail)
                session.commit()
        return acc_status_detail

    def create_by_self(self, is_new_day=False):
        """
        创建新的对象，默认前一日持仓信息的最新价，等于下一交易日的结算价（即 AvePrice）
        :return: 
        """
        cash_available_last_day = self.cash_available if is_new_day else self.cash_available_last_day
        detail = TradeAgentStatusDetail(stg_run_id=self.stg_run_id,
                                        trade_agent_key=self.trade_agent_key,
                                        trade_date=self.trade_date,
                                        trade_time=self.trade_time,
                                        trade_millisec=self.trade_millisec,
                                        cash_available_last_day=cash_available_last_day,
                                        cash_available=self.cash_available,
                                        curr_margin=self.curr_margin,
                                        close_profit=self.close_profit,
                                        position_profit=self.position_profit,
                                        floating_pl_cum=self.floating_pl_cum,
                                        cashflow_daily=self.cashflow_daily,
                                        cashflow_cum=self.cashflow_cum,
                                        commission_tot=self.commission_tot,
                                        cash_init=self.cash_init,
                                        calc_mode=self.calc_mode,
                                        run_mode=self.run_mode,
                                        )
        detail.last_status = self
        return detail

    def update_by_pos_status_detail(
            self, pos_status_detail_dic, timestamp_curr: (datetime, pd.Timestamp) = None, md: dict = None,
            timestamp_key=None, date_key=None, time_key=None, milli_sec_key=None):
        """
        根据 持仓列表更新账户信息
        :param pos_status_detail_dic:
        :param timestamp_curr: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param md: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param timestamp_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param date_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param time_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param milli_sec_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :return:
        """
        if self.run_mode == RunMode.Backtest.value:
            detail = self._update_by_pos_status_detail(
                pos_status_detail_dic, timestamp_curr, md, timestamp_key, date_key, time_key, milli_sec_key)
        elif self.run_mode == RunMode.Backtest_FixPercent.value:
            detail = self._update_by_pos_status_detail_fix_percent(
                pos_status_detail_dic, timestamp_curr, md, timestamp_key, date_key, time_key, milli_sec_key)

        detail.pos_status_detail_dic = pos_status_detail_dic.copy()
        return detail

    def _update_by_pos_status_detail(self, pos_status_detail_dic, timestamp_curr: (datetime, pd.Timestamp) = None,
                                     md: dict = None,
                                     timestamp_key=None, date_key=None, time_key=None, milli_sec_key=None):
        """
        根据 持仓列表更新账户信息
        :param pos_status_detail_dic:
        :param timestamp_curr: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param md: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param timestamp_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param date_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param time_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param milli_sec_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :return:
        """
        # warnings.warn('该函数为范例函数，可以根据实际情况改写', UserWarning)
        # 更新日期、时间
        if timestamp_curr is None:
            timestamp_curr = md[timestamp_key]

        if timestamp_curr is None:
            trade_date = str_2_date(md[date_key]) - timedelta(days=1)
            trade_time = pd_timedelta_2_timedelta(md[time_key])
            trade_dt = datetime_2_str(date_time_2_str(trade_date, trade_time))
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0
        else:
            trade_date = timestamp_curr.date()
            trade_time = timestamp_curr.time()
            trade_dt = timestamp_curr
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0

        is_new_day = self.trade_date != trade_date
        detail = self.create_by_self(is_new_day=is_new_day)
        detail.trade_dt = trade_dt
        detail.trade_date = trade_date
        detail.trade_time = trade_time
        detail.trade_millisec = trade_millisec

        # 更新非日期数据
        curr_margin = 0
        position_value = 0
        close_profit = 0
        position_profit = 0
        floating_pl_cum = 0
        cashflow_daily, cashflow_cum = 0, 0
        commission_tot = 0
        for instrument_id, pos_status_detail in pos_status_detail_dic.items():
            position_value += pos_status_detail.position_value
            curr_margin += pos_status_detail.margin
            position_profit += pos_status_detail.floating_pl
            close_profit += (pos_status_detail.floating_pl_cum - pos_status_detail.floating_pl)
            floating_pl_cum += pos_status_detail.floating_pl_cum
            cashflow_daily += pos_status_detail.cashflow_daily
            cashflow_cum += pos_status_detail.cashflow_cum
            commission_tot += pos_status_detail.commission_tot

        cash_available = self.cash_init + cashflow_cum

        # 记录当前可用现金
        detail.cash_available = cash_available
        cash_available_last_day = detail.cash_available_last_day
        calc_gap = cash_available - (cash_available_last_day + cashflow_daily)
        if calc_gap < -0.01 or 0.01 < calc_gap:
            self.logger.warning(
                "%s cash_init + cashflow_cum = %10.2f 比 cash_available_last_day + cashflow_daily = %10.2f 多 %10.2f。"
                "cash_init=%10.2f cashflow_cum=%10.2f cash_available_last_day=%10.2f cashflow_daily=%10.2f",
                detail.trade_dt,
                cash_available, cash_available_last_day + cashflow_daily,
                                cash_available - (cash_available_last_day + cashflow_daily),
                self.cash_init, cashflow_cum, cash_available_last_day, cashflow_daily)
        else:
            pass

        detail.position_value = position_value
        detail.curr_margin = curr_margin
        detail.close_profit = close_profit
        detail.position_profit = position_profit
        detail.floating_pl_cum = floating_pl_cum
        detail.cashflow_daily = cashflow_daily
        detail.cashflow_cum = cashflow_cum
        detail.commission_tot = commission_tot
        detail.cash_and_margin = detail.cash_available + curr_margin
        detail.rr = detail.floating_pl_cum / detail.cash_init
        detail.rr_nc = (detail.floating_pl_cum + commission_tot) / detail.cash_init
        # 当期盈利及现金的计算均是按照单利计算的，计算rr的时候需要将单利转化为复利
        # 计算方法为：
        # 当期状态复利rr = ("当期状态的单利 rr" - "上一状态的单利 rr" + 1) * ("上一状态的复利 rr" + 1) - 1
        # 进一步合并公式 =  （“当期 cash_and_margin” - “上一状态 cash_and_margin” + cash_init）/ cash_init * "上一状态的复利 rr"
        detail.rr_compound = (detail.rr - self.rr + 1) * (self.rr + 1) - 1
        detail.rr_compound_nc = (detail.rr_nc - self.rr_nc + 1) * (self.rr_nc + 1) - 1

        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()
        return detail

    def _update_by_pos_status_detail_fix_percent(self, pos_status_detail_dic,
                                                 timestamp_curr: (datetime, pd.Timestamp) = None, md: dict = None,
                                                 timestamp_key=None, date_key=None, time_key=None, milli_sec_key=None):
        """
        根据 持仓列表更新账户信息
        :param pos_status_detail_dic:
        :param timestamp_curr: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param md: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param timestamp_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param date_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param time_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :param milli_sec_key: timestamp_curr 或 md + key 参数 两组参数二选其一
        :return:
        """
        # warnings.warn('该函数为范例函数，可以根据实际情况改写', UserWarning)
        # 更新日期、时间
        if timestamp_curr is None:
            timestamp_curr = md[timestamp_key]

        if timestamp_curr is None:
            trade_date = str_2_date(md[date_key]) - timedelta(days=1)
            trade_time = pd_timedelta_2_timedelta(md[time_key])
            trade_dt = datetime_2_str(date_time_2_str(trade_date, trade_time))
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0
        else:
            trade_date = timestamp_curr.date()
            trade_time = timestamp_curr.time()
            trade_dt = timestamp_curr
            trade_millisec = int(md.setdefault(milli_sec_key, 0)) if milli_sec_key is not None else 0

        is_new_day = self.trade_date != trade_date
        detail = self.create_by_self(is_new_day=is_new_day)
        detail.trade_dt = trade_dt
        detail.trade_date = trade_date
        detail.trade_time = trade_time
        detail.trade_millisec = trade_millisec

        # 上一状态时的 pos_status_detail_dic
        # pos_status_detail_dic_last = self.last_status.pos_status_detail_dic if self.last_status is not None else {}
        pos_status_detail_dic_last = self.pos_status_detail_dic
        # 更新非日期数据
        curr_margin = 0
        position_value = 0
        close_profit = 0
        position_profit = 0
        floating_pl_cum = 0
        cashflow_daily, cashflow_cum = 0, 0
        commission_tot = 0
        position_rate_tot = 0
        for instrument_id, pos_status_detail in pos_status_detail_dic.items():
            position_rate = pos_status_detail.position
            curr_margin += position_rate  # curr_margin 记录的是持仓比例例如 1 满仓; 0 空仓;
            position_rate_tot += position_rate  # 累计持仓比例 1 满仓; 0 空仓;
            # 固定比例仓位下，固定的是“margin 固定保证金比例”，而不是 “position_value 持仓手数”，因此，其他数据的计算需要进行一次转换
            # margin_2_pos_rate 是用来记录 从在固定 margin 的情况下，其他数据的转换比例
            if instrument_id in pos_status_detail_dic_last:
                pos_status_detail_last = pos_status_detail_dic_last[instrument_id]
                if pos_status_detail.margin > 0:
                    margin_2_pos_rate = position_rate / pos_status_detail.margin
                elif pos_status_detail.margin == 0 and pos_status_detail_last.margin > 0:
                    margin_2_pos_rate = position_rate / pos_status_detail_last.margin
                else:
                    margin_2_pos_rate = 0

                # 计算 close_profit, floating_pl_cum, cashflow_cum, commission_tot 需要与上一状态插值进行计算
                close_profit += ((pos_status_detail.floating_pl_cum - pos_status_detail.floating_pl) -
                                 (pos_status_detail_last.floating_pl_cum - pos_status_detail_last.floating_pl)
                                 ) * margin_2_pos_rate
                floating_pl_cum += (pos_status_detail.floating_pl_cum - pos_status_detail_last.floating_pl_cum
                                    ) * margin_2_pos_rate
                cashflow_cum += (pos_status_detail.cashflow_cum - pos_status_detail_last.cashflow_cum
                                 ) * margin_2_pos_rate
                commission_tot += (pos_status_detail.commission_tot - pos_status_detail_last.commission_tot
                                   ) * margin_2_pos_rate

            else:
                if pos_status_detail.margin > 0:
                    margin_2_pos_rate = position_rate / pos_status_detail.margin
                else:
                    margin_2_pos_rate = 0

                # 计算 close_profit, floating_pl_cum, cashflow_cum, commission_tot 需要与上一状态插值进行计算
                close_profit += (pos_status_detail.floating_pl_cum - pos_status_detail.floating_pl) * margin_2_pos_rate
                floating_pl_cum += pos_status_detail.floating_pl_cum * margin_2_pos_rate
                cashflow_cum += pos_status_detail.cashflow_cum * margin_2_pos_rate
                commission_tot += pos_status_detail.commission_tot * margin_2_pos_rate

            # 计算position_value, cashflow_daily, position_profit
            position_value += pos_status_detail.position_value * margin_2_pos_rate
            cashflow_daily += pos_status_detail.cashflow_daily * margin_2_pos_rate
            position_profit += pos_status_detail.floating_pl * margin_2_pos_rate

        # 计算 close_profit, floating_pl_cum, cashflow_cum, commission_tot 需要与上一状态插值进行计算
        close_profit = self.close_profit + close_profit
        floating_pl_cum = self.floating_pl_cum + floating_pl_cum
        cashflow_cum = self.cashflow_cum + cashflow_cum
        commission_tot = self.commission_tot + commission_tot

        # 这里不考虑手续费造成的实际可用现金可能为负数的清空，仅简单计算 cash + margin = 1
        # cash_available = 1 - curr_margin

        # 记录当前可用现金，在补丁比例模式下，可用现金等于初始现金 - 累计持仓比例
        detail.cash_available = cash_available = self.cash_init - position_rate_tot
        # 以下检查不适用。因为固定比例持仓分析，cash_available 相当于闲置现金比例，不会随盈利现金增长而等比例变化
        # cash_available_last_day = detail.cash_available_last_day
        # calc_gap = cash_available - (cash_available_last_day + cashflow_daily)
        # if calc_gap < -0.01 or 0.01 < calc_gap:
        #     self.logger.warning(
        #         "%s cash_init + cashflow_cum = %10.5f 比 cash_available_last_day + cashflow_daily = %10.5f 多 %10.5f。"
        #         "cash_init=%10.5f cashflow_cum=%10.5f cash_available_last_day=%10.5f cashflow_daily=%10.5f",
        #         detail.trade_dt,
        #         cash_available, cash_available_last_day + cashflow_daily,
        #                         cash_available - (cash_available_last_day + cashflow_daily),
        #         self.cash_init, cashflow_cum, cash_available_last_day, cashflow_daily)

        detail.position_value = position_value
        detail.curr_margin = curr_margin
        detail.close_profit = close_profit
        detail.position_profit = position_profit
        detail.floating_pl_cum = floating_pl_cum
        detail.cashflow_daily = cashflow_daily
        detail.cashflow_cum = cashflow_cum
        detail.commission_tot = commission_tot
        detail.cash_and_margin = self.cash_init + cashflow_cum + curr_margin
        detail.rr = detail.floating_pl_cum / detail.cash_init
        detail.rr_nc = (detail.floating_pl_cum + commission_tot) / detail.cash_init
        # 当期盈利及现金的计算均是按照单利计算的，计算rr的时候需要将单利转化为复利
        # 计算方法为：
        # 当期状态复利rr = ("当期状态的单利 rr" - "上一状态的单利 rr" + 1) * ("上一状态的复利 rr" + 1) - 1
        # 进一步合并公式 =  （“当期 cash_and_margin” - “上一状态 cash_and_margin” + cash_init）/ cash_init * "上一状态的复利 rr"
        detail.rr_compound = (detail.rr - self.rr + 1) * (self.rr + 1) - 1
        detail.rr_compound_nc = (detail.rr_nc - self.rr_nc + 1) * (self.rr_nc + 1) - 1

        if config.ORM_UPDATE_OR_INSERT_PER_ACTION:
            # 更新最新持仓纪录
            with with_db_session(engine_ibats, expire_on_commit=False) as session:
                session.add(detail)
                session.commit()
        return detail


class StgRunStatusDetail(BaseModel):
    """记录每一单位时刻 所有trade_agent_detail 状态汇总"""
    __tablename__ = 'stg_run_status_detail'
    stg_run_id = Column(Integer, primary_key=True)  # 对应回测了策略 StgRunID 此数据与 AccSumID 对应数据相同
    stg_run_status_detail_idx = Column(Integer, primary_key=True)
    trade_dt = Column(DateTime)
    trade_date = Column(Date)  # 对应行情数据中 ActionDate
    trade_time = Column(Time)  # 对应行情数据中 ActionTime
    trade_millisec = Column(Integer)  # 对应行情数据中 ActionMillisec
    cash_available_last_day = Column(DOUBLE, default=0.0)  # 可用资金, double
    cash_available = Column(DOUBLE, default=0.0)  # 可用资金, double
    position_value = Column(DOUBLE, default=0.0)
    curr_margin = Column(DOUBLE, default=0.0)  # 当前保证金总额, double
    close_profit = Column(DOUBLE, default=0.0)
    position_profit = Column(DOUBLE, default=0.0)
    floating_pl_cum = Column(DOUBLE, default=0.0)
    commission_tot = Column(DOUBLE, default=0.0)
    cash_init = Column(DOUBLE, default=0.0)
    cash_and_margin = Column(DOUBLE, default=0.0)
    cashflow_daily = Column(DOUBLE, default=0.0)
    cashflow_cum = Column(DOUBLE, default=0.0)
    rr = Column(DOUBLE, default=0.0)  # Return Rate
    rr_nc = Column(DOUBLE, default=0.0)  # Return Rate Compound
    rr_compound = Column(DOUBLE, default=0.0)  # Return Rate No Commission
    rr_compound_nc = Column(DOUBLE, default=0.0)  # Return Rate Compound No Commission

    def __init__(self, stg_run_id=None,
                 trade_dt=None, trade_date=None, trade_time=None, trade_millisec=None, cash_available_last_day=0.0,
                 cash_available=None, curr_margin=None, close_profit=None, position_profit=None, floating_pl_cum=None,
                 commission_tot=None, cash_init=None, cashflow_daily=0.0, cashflow_cum=0.0,
                 rr=0, rr_nc=0.0, rr_compound=0.0, rr_compound_nc=0.0):
        self.stg_run_id = stg_run_id
        self.stg_run_status_detail_idx = None if stg_run_id is None else idx_generator(
            stg_run_id, StgRunStatusDetail)
        self.trade_dt = date_time_2_str(trade_date, trade_time) if trade_dt is None else trade_dt
        self.trade_date = trade_date
        self.trade_time = trade_time
        self.trade_millisec = trade_millisec
        self.cash_available_last_day = cash_available_last_day
        self.cash_available = cash_available
        self.curr_margin = curr_margin
        self.close_profit = close_profit
        self.position_profit = position_profit
        self.floating_pl_cum = floating_pl_cum
        self.commission_tot = commission_tot
        self.cash_init = cash_init
        self.cash_and_margin = cash_available + curr_margin
        self.cashflow_daily = cashflow_daily
        self.cashflow_cum = cashflow_cum
        self.rr = rr
        self.rr_nc = rr_nc
        self.rr_compound = rr_compound
        self.rr_compound_nc = rr_compound_nc

    @staticmethod
    def create_t_1(
            stg_run_id, trade_agent_status_detail_list):
        """通过 trade_agent_status_detail_list 更新当前 stg_run_detail 状态"""
        data_len = len(trade_agent_status_detail_list)
        if data_len == 0:
            return None

        trade_dt, trade_date, trade_time, trade_millisec = None, None, None, None
        cash_available_last_day, cash_available, curr_margin, close_profit, position_profit = 0, 0, 0, 0, 0
        floating_pl_cum, commission_tot, cash_init, cash_and_margin, cashflow_daily, cashflow_cum, rr = 0, 0, 0, 0, 0, 0, 0
        for detail in trade_agent_status_detail_list:
            if trade_dt is None or trade_dt < detail.trade_dt:
                trade_dt = detail.trade_dt
                trade_date = detail.trade_date
                trade_time = detail.trade_time
                trade_millisec = detail.trade_millisec

            cash_available_last_day += 0 if detail.cash_available_last_day is None else detail.cash_available_last_day
            cash_available += 0 if detail.cash_available is None else detail.cash_available
            curr_margin += 0 if detail.curr_margin is None else detail.curr_margin
            close_profit += 0 if detail.close_profit is None else detail.close_profit
            position_profit += 0 if detail.position_profit is None else detail.position_profit
            floating_pl_cum += 0 if detail.floating_pl_cum is None else detail.floating_pl_cum
            commission_tot += 0 if detail.commission_tot is None else detail.commission_tot
            cash_init += 0 if detail.cash_init is None else detail.cash_init
            cash_and_margin += 0 if detail.cash_and_margin is None else detail.cash_and_margin
            cashflow_daily += 0 if detail.cashflow_daily is None else detail.cashflow_daily
            cashflow_cum += 0 if detail.cashflow_cum is None else detail.cashflow_cum

        detail = StgRunStatusDetail(
            stg_run_id=stg_run_id,
            trade_dt=trade_dt,
            trade_date=trade_date,
            trade_time=trade_time,
            trade_millisec=trade_millisec,
            cash_available_last_day=cash_available_last_day,
            cash_available=cash_available,
            curr_margin=curr_margin,
            close_profit=close_profit,
            position_profit=position_profit,
            floating_pl_cum=floating_pl_cum,
            commission_tot=commission_tot,
            cash_init=cash_init,
            cashflow_daily=cashflow_daily,
            cashflow_cum=cashflow_cum,
        )

        return detail

    def update_by_trade_agent_status_detail_list(self, trade_agent_status_detail_list):
        data_len = len(trade_agent_status_detail_list)
        if data_len == 0:
            return None

        trade_dt, trade_date, trade_time, trade_millisec = None, None, None, None
        cash_available_last_day, cash_available, curr_margin, close_profit, position_profit = 0, 0, 0, 0, 0
        floating_pl_cum, commission_tot, cash_init, cash_and_margin, cashflow_daily, cashflow_cum, rr = 0, 0, 0, 0, 0, 0, 0
        for detail in trade_agent_status_detail_list:
            if trade_dt is None or trade_dt < detail.trade_dt:
                trade_dt = detail.trade_dt
                trade_date = detail.trade_date
                trade_time = detail.trade_time
                trade_millisec = detail.trade_millisec

            cash_available_last_day += 0 if detail.cash_available_last_day is None else detail.cash_available_last_day
            cash_available += 0 if detail.cash_available is None else detail.cash_available
            curr_margin += 0 if detail.curr_margin is None else detail.curr_margin
            close_profit += 0 if detail.close_profit is None else detail.close_profit
            position_profit += 0 if detail.position_profit is None else detail.position_profit
            floating_pl_cum += 0 if detail.floating_pl_cum is None else detail.floating_pl_cum
            commission_tot += 0 if detail.commission_tot is None else detail.commission_tot
            cash_init += 0 if detail.cash_init is None else detail.cash_init
            cash_and_margin += 0 if detail.cash_and_margin is None else detail.cash_and_margin
            cashflow_daily += 0 if detail.cashflow_daily is None else detail.cashflow_daily
            cashflow_cum += 0 if detail.cashflow_cum is None else detail.cashflow_cum

        rr = floating_pl_cum / cash_init
        rr_nc = (floating_pl_cum + commission_tot) / cash_init
        # 当期盈利及现金的计算均是按照单利计算的，计算rr的时候需要将单利转化为复利
        # 计算方法为：
        # 当期状态复利rr = ("当期状态的单利 rr" - "上一状态的单利 rr" + 1) * ("上一状态的复利 rr" + 1) - 1
        # 进一步合并公式 =  （“当期 cash_and_margin” - “上一状态 cash_and_margin” + cash_init）/ cash_init * "上一状态的复利 rr"
        rr_compound = (rr - self.rr + 1) * (self.rr + 1) - 1
        rr_compound_nc = (rr_nc - self.rr_nc + 1) * (self.rr_nc + 1) - 1

        detail = StgRunStatusDetail(
            stg_run_id=self.stg_run_id,
            trade_dt=trade_dt,
            trade_date=trade_date,
            trade_time=trade_time,
            trade_millisec=trade_millisec,
            cash_available_last_day=cash_available_last_day,
            cash_available=cash_available,
            curr_margin=curr_margin,
            close_profit=close_profit,
            position_profit=position_profit,
            floating_pl_cum=floating_pl_cum,
            commission_tot=commission_tot,
            cash_init=cash_init,
            cashflow_daily=cashflow_daily,
            cashflow_cum=cashflow_cum,
            rr=rr,
            rr_nc=rr_nc,
            rr_compound=rr_compound,
            rr_compound_nc=rr_compound_nc,
        )
        return detail


class HeartBeat(BaseModel):
    """心跳信息表"""

    __tablename__ = 'heart_beat'
    update_dt = Column(DateTime, primary_key=True)


def init_data():
    from sqlalchemy.sql import func
    with with_db_session(engine_ibats) as session:
        count = session.query(func.count(HeartBeat.update_dt)).scalar()
        if count == 0:
            logger.debug('初始化 HeartBeat 数据')
            session.add(HeartBeat(update_dt=datetime.now()))
            session.commit()
        else:
            session.query(HeartBeat).update({HeartBeat.update_dt: datetime.now()})
            session.commit()


def start_heart_beat_thread():
    """该方法已废弃"""
    import warnings
    warnings.filterwarnings(
        'default', '该方法以及废弃，有SqlAlchemy中的 CreateEngine pool_pre_ping=True 功能替代，'
                   '详见 IBATS_Utils 2020-01-05 commit 8f56a87f')
    import time
    global _HEART_BEAT_THREAD
    beat_logger = logging.getLogger('heart_beat')
    if _HEART_BEAT_THREAD is not None and _HEART_BEAT_THREAD.is_alive():
        beat_logger.info('timer_heart_beat thread is running')
        return

    def timer_heart_beat():
        beat_logger.debug('timer_heart_beat thread start')
        is_debug, n = False, 10
        while True:
            if is_debug:
                if n <= 0:
                    break
                else:
                    n -= 1

            time.sleep(1800)
            try:
                with with_db_session(engine_ibats) as session:
                    update_dt = datetime.now()
                    session.query(HeartBeat).update({HeartBeat.update_dt: update_dt})
                    session.commit()
                beat_logger.debug('heart beat at %s', update_dt)
            except:
                beat_logger.exception('heart beat exception')
                break

        beat_logger.debug('timer_heart_beat thread finished')

    import threading
    _HEART_BEAT_THREAD = threading.Thread(target=timer_heart_beat)
    _HEART_BEAT_THREAD.daemon = True
    _HEART_BEAT_THREAD.start()


def init():
    from ibats_utils.db import alter_table_2_myisam
    global engine_ibats
    engine_ibats = engines.engine_ibats
    BaseModel.metadata.create_all(engine_ibats)
    alter_table_2_myisam(engine_ibats)
    logger.info("所有表结构建立完成")
    init_data()


# 该方法已废弃
# start_heart_beat_thread()


if __name__ == "__main__":
    init()
    # 创建user表，继承metadata类
    # Engine使用Schama Type创建一个特定的结构对象
    # stg_info_table = Table("stg_info", metadata, autoload=True)
