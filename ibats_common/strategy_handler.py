#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:47
@File    : strategy_handler.py
@contact : mmmaaaggg@163.com
@desc    : 策略处理句柄，用于处理策略进行回测或实盘交易
"""
import json
import logging
from threading import Thread
import warnings
import numpy as np
from queue import Empty
import time
from datetime import date, datetime, timedelta
from abc import ABC
import pandas as pd
from ibats_common.backend import engines
from ibats_common.backend.orm import StgRunInfo
from ibats_common.common import ExchangeName, RunMode, ContextKey, PeriodType
from ibats_common.md import md_agent_factory
from ibats_common.strategy import StgBase
from ibats_common.utils.db import with_db_session
from ibats_common.utils.mess import try_2_date
from ibats_common.trade import trader_agent_factory

engine_ibats = engines.engine_ibats
logger_stg_base = logging.getLogger(__name__)


class StgHandlerBase(Thread, ABC):

    def __init__(self, stg_run_id, stg_base: StgBase, run_mode, md_period_agent_dic):
        super().__init__(daemon=True)
        self.stg_run_id = stg_run_id
        self.run_mode = run_mode
        # 初始化策略实体，传入参数
        self.stg_base = stg_base
        # 设置工作状态
        self.is_working = False
        self.is_done = False
        # 日志
        self.logger = logging.getLogger()
        # 对不同周期设置相应的md_agent
        self.md_period_agent_dic = md_period_agent_dic

    def stg_run_ending(self):
        """
        处理策略结束相关事项
        释放策略资源
        更新策略执行信息
        :return:
        """
        self.stg_base.release()
        # 更新数据库 td_to 字段
        with with_db_session(engine_ibats) as session:
            session.query(StgRunInfo).filter(StgRunInfo.stg_run_id == self.stg_run_id).update(
                {StgRunInfo.dt_to: datetime.now()})
            # sql_str = StgRunInfo.update().where(
            # StgRunInfo.c.stg_run_id == self.stg_run_id).values(dt_to=datetime.now())
            # session.execute(sql_str)
            session.commit()

        self.is_working = False
        self.is_done = True

    def __repr__(self):
        return '<{0.__class__.__name__}:{0.stg_run_id} {0.run_mode}>'.format(self)


class StgHandlerRealtime(StgHandlerBase):

    def __init__(self, stg_run_id, stg_base: StgBase, md_period_agent_dic, **kwargs):
        super().__init__(stg_run_id=stg_run_id, stg_base=stg_base, run_mode=RunMode.Realtime,
                         md_period_agent_dic=md_period_agent_dic)
        # 对不同周期设置相应的md_agent
        self.md_period_agent_dic = md_period_agent_dic
        # 设置线程池
        self.running_thread = {}
        # 日志
        self.logger = logging.getLogger()
        # 设置推送超时时间
        self.timeout_pull = 60
        # 设置独立的时间线程
        self.enable_timer_thread = kwargs.setdefault('enable_timer_thread', False)
        self.seconds_of_timer_interval = kwargs.setdefault('seconds_of_timer_interval', 9999)

    def run(self):

        # TODO: 以后再加锁，防止多线程，目前只是为了防止误操作导致的重复执行
        if self.is_working:
            return
        else:
            self.is_working = True

        try:
            # 策略初始化
            self.stg_base.init()
            # 对各个周期分别设置对应 handler
            for period, md_agent in self.md_period_agent_dic.items():
                # 获取对应事件响应函数
                on_period_md_handler = self.stg_base.on_period_md_handler
                # 异步运行：每一个周期及其对应的 handler 作为一个线程独立运行
                thread_name = 'run_md_agent %s' % md_agent.name
                run_md_agent_thread = Thread(target=self.run_md_agent, name=thread_name,
                                             args=(md_agent, on_period_md_handler), daemon=True)
                self.running_thread[period] = run_md_agent_thread
                self.logger.info("加载 %s 线程", thread_name)
                run_md_agent_thread.start()

            if self.enable_timer_thread:
                thread_name = 'run_timer'
                timer_thread = Thread(target=self.run_timer, name=thread_name, daemon=True)
                self.logger.info("加载 %s 线程", thread_name)
                timer_thread.start()

            # 各个线程分别join等待结束信号
            for period, run_md_agent_thread in self.running_thread.items():
                run_md_agent_thread.join()
                self.logger.info('%s period %s finished', run_md_agent_thread.name, period)
        finally:
            self.is_working = False
            self.stg_run_ending()

    def run_timer(self):
        """
        负责定时运行策略对象的 on_timer 方法
        :return:
        """
        while self.is_working:
            try:
                self.stg_base.on_timer()
            except:
                self.logger.exception('on_timer 函数运行异常')
            finally:
                time.sleep(self.seconds_of_timer_interval)

    def run_md_agent(self, md_agent, handler):
        """
        md_agent pull 方法的事件驱动处理函数
        :param md_agent:
        :param handler:  self.stgbase对象的响应 md_agent 的梳理函数：根据不同的 md_period 可能是
         on_tick、 on_min、 on_day、 on_week、 on_month 等其中一个
        :return:
        """
        period = md_agent.md_period
        self.logger.info('启动 %s 行情监听线程', period)
        md_agent.connect()
        md_agent.subscribe()  # 参数为空相当于 md_agent.subscribe(md_agent.instrument_id_list)
        md_agent.start()
        md_dic = None
        while self.is_working:
            try:
                if not self.is_working:
                    break
                # 加载数据，是设置超时时间，防止长时间阻塞
                md_dic = md_agent.pull(self.timeout_pull)
                handler(period, md_dic)
            except Empty:
                # 工作状态检查
                pass
            except Exception:
                self.logger.exception('%s 事件处理句柄执行异常，对应行情数据md_dic:\n%s',
                                      period, md_dic)
                # time.sleep(1)
        md_agent.release()
        self.logger.info('period:%s finished', period)


class StgHandlerBacktest(StgHandlerBase):

    def __init__(self, stg_run_id, stg_base: StgBase, md_period_agent_dic, date_from, date_to, **kwargs):
        super().__init__(stg_run_id=stg_run_id, stg_base=stg_base, run_mode=RunMode.Backtest,
                         md_period_agent_dic=md_period_agent_dic)
        # 回测 ID 每一次测试生成一个新的ID，在数据库中作为本次测试的唯一标识
        # TODO: 这一ID需要从数据库生成
        # self.backtest_id = 1
        # self.stg_base._trade_agent.backtest_id = self.backtest_id
        # 设置回测时间区间
        self.date_from = try_2_date(date_from)
        self.date_to = try_2_date(date_to)
        if not isinstance(self.date_from, date):
            raise ValueError("date_from: %s", date_from)
        if not isinstance(self.date_to, date):
            raise ValueError("date_from: %s", date_to)
        # 初始资金账户金额
        self.init_cash = kwargs['init_cash']
        # 载入回测时间段各个周期的历史数据，供回测使用
        # 对各个周期分别进行处理
        self.backtest_his_df_dic = {}
        for period, md_agent in self.md_period_agent_dic.items():
            md_df = md_agent.load_history(date_from, date_to, load_md_count=0)
            if md_df is None:
                continue
            if isinstance(md_df, pd.DataFrame):
                # 对于 CTP 老程序接口直接返回的是 df，因此补充相关的 key 数据
                # TODO: 未来这部分代码将逐步给更替
                warnings.warn('load_history 需要返回 dict 类型数据， 对 DataFame 的数据处理即将废弃', DeprecationWarning)
                if period == PeriodType.Tick:
                    his_df_dic = {'md_df': md_df,
                                  'date_key': 'ActionDay', 'time_key': 'ActionTime',
                                  'microseconds_key': 'ActionMillisec'}
                else:
                    his_df_dic = {'md_df': md_df,
                                  'date_key': 'ActionDay', 'time_key': 'ActionTime'}
                self.backtest_his_df_dic[period] = his_df_dic
                self.logger.debug('加载 %s 回测数据 %d 条记录', period, md_df.shape[0])
            else:
                self.backtest_his_df_dic[period] = his_df_dic = md_df
                self.logger.debug('加载 %s 回测数据 %d 条记录', period, his_df_dic['md_df'].shape[0])

    def run(self):
        """
        执行回测
        :return:
        """
        # TODO: 以后再加锁，防止多线程，目前只是为了防止误操作导致的重复执行
        if self.is_working:
            self.logger.warning('当前任务正在执行中..，避免重复执行')
            return
        else:
            self.is_working = True
        self.logger.info('执行回测任务【%s - %s】开始', self.date_from, self.date_to)
        try:
            # 策略初始化
            self.stg_base.init()
            # 对每一个周期构建时间轴及对应记录的数组下标
            period_dt_idx_dic = {}
            for period, his_df_dic in self.backtest_his_df_dic.items():
                his_df = his_df_dic['md_df']
                df_len = his_df.shape[0]
                if df_len == 0:
                    continue
                datetime_s = his_df[his_df_dic['datetime_key']] if 'datetime_key' in his_df_dic else None
                date_s = his_df[his_df_dic['date_key']] if 'date_key' in his_df_dic else None
                time_s = his_df[his_df_dic['time_key']] if 'time_key' in his_df_dic else None
                microseconds_s = his_df[his_df_dic['microseconds_key']] if 'microseconds_key' in his_df_dic else None
                # 整理日期轴
                dt_idx_dic = {}
                if datetime_s is not None:
                    for idx in range(df_len):
                        if microseconds_s:
                            dt = datetime_s[idx] + timedelta(microseconds=int(microseconds_s[idx]))
                        else:
                            dt = datetime_s[idx]

                        if dt in dt_idx_dic:
                            dt_idx_dic[dt].append(idx)
                        else:
                            dt_idx_dic[dt] = [idx]
                elif date_s is not None and time_s is not None:
                    for idx in range(df_len):
                        # action_date = date_s[idx]
                        # dt = datetime(action_date.year, action_date.month, action_date.day) + time_s[
                        #     idx] + timedelta(microseconds=int(microseconds_s[idx]))
                        if microseconds_s:
                            dt = datetime.combine(date_s[idx], time_s[idx]) + timedelta(
                                microseconds=int(microseconds_s[idx]))
                        else:
                            dt = datetime.combine(date_s[idx], time_s[idx])

                        if dt in dt_idx_dic:
                            dt_idx_dic[dt].append(idx)
                        else:
                            dt_idx_dic[dt] = [idx]

                # action_day_s = his_df['ActionDay']
                # action_time_s = his_df['ActionTime']
                # # Tick 数据 存在 ActionMillisec 记录秒以下级别数据
                # if period == PeriodType.Tick:
                #     action_milsec_s = his_df['ActionMillisec']
                #     dt_idx_dic = {}
                #     for idx in range(df_len):
                #         action_date = action_day_s[idx]
                #         dt = datetime(action_date.year, action_date.month, action_date.day) + action_time_s[
                #             idx] + timedelta(microseconds=int(action_milsec_s[idx]))
                #         if dt in dt_idx_dic:
                #             dt_idx_dic[dt].append(idx)
                #         else:
                #             dt_idx_dic[dt] = [idx]
                # else:
                #     dt_idx_dic = {}
                #     for idx in range(df_len):
                #         action_date = action_day_s[idx]
                #         dt = datetime(action_date.year, action_date.month, action_date.day) + action_time_s[
                #             idx]
                #         if dt in dt_idx_dic:
                #             dt_idx_dic[dt].append(idx)
                #         else:
                #             dt_idx_dic[dt] = [idx]
                # 记录各个周期时间戳
                period_dt_idx_dic[period] = dt_idx_dic

                # 设置各个周期相关 key 给 trade_agent
                if 'datetime_key' in his_df_dic:
                    self.stg_base.trade_agent.set_timestamp_key(his_df_dic['datetime_key'])
                if 'symbol_key' in his_df_dic:
                    self.stg_base.trade_agent.set_symbol_key(his_df_dic['symbol_key'])
                if 'close_key' in his_df_dic:
                    self.stg_base.trade_agent.set_close_key(his_df_dic['close_key'])

            # 按照时间顺序将各个周期数据依次推入对应 handler
            period_idx_df = pd.DataFrame(period_dt_idx_dic).sort_index()
            for row_num in range(period_idx_df.shape[0]):
                period_idx_s = period_idx_df.ix[row_num, :]
                for period, idx_list in period_idx_s.items():
                    if all(np.isnan(idx_list)):
                        continue
                    his_df = self.backtest_his_df_dic[period]['md_df']
                    for idx_row in idx_list:
                        # TODO: 这里存在着性能优化空间 DataFrame -> Series -> dict 效率太低
                        md = his_df.ix[idx_row].to_dict()
                        # 在回测阶段，需要对 trade_agent 设置最新的md数据，一遍交易接口确认相应的k线日期
                        self.stg_base.trade_agent.set_curr_md(period, md)
                        # 执行策略相应的事件响应函数
                        self.stg_base.on_period_md_handler(period, md)
                        # 根据最新的 md 及 持仓信息 更新 账户信息
                        self.stg_base.trade_agent.update_account_info()
            self.logger.info('执行回测任务【%s - %s】完成', self.date_from, self.date_to)
        finally:
            self.is_working = False
            self.stg_run_ending()


def strategy_handler_factory(stg_class: type(StgBase), strategy_params, md_agent_params_list, run_mode: RunMode,
                             exchange_name: ExchangeName, **trade_agent_params) -> StgHandlerBase:
    """
    单一交易所策略处理具备
    建立策略对象
    建立数据库相应记录信息
    根据运行模式（实时、回测）：选择相应的md_agent以及trade_agent
    :param stg_class: 策略类型 StgBase 的子类
    :param strategy_params: 策略参数
    :param md_agent_params_list: 行情代理（md_agent）参数，支持同时订阅多周期、多品种，
    例如：同时订阅 [ethusdt, eosusdt] 1min 行情、[btcusdt, ethbtc] tick 行情
    :param exchange_name: 选择交易所接口 ExchangeName
    :param run_mode: 运行模式 RunMode.Realtime  或 RunMode.Backtest
    :param trade_agent_params: 运行参数，回测模式下：运行起止时间，实时行情下：加载定时器等设置
    :return: 策略执行对象实力
    """
    md_agent_params_list_4_json = []
    for md_agent_param in md_agent_params_list:
        md_agent_param_4_json = md_agent_param.copy()
        md_agent_param_4_json['exchange_name'] = exchange_name.name
        md_agent_param_4_json['run_mode'] = run_mode.name
        md_agent_params_list_4_json.append(md_agent_param_4_json)

        md_agent_param['exchange_name'] = exchange_name
        md_agent_param['run_mode'] = run_mode

    trade_agent_params_4_json = trade_agent_params.copy()
    trade_agent_params_4_json['exchange_name'] = exchange_name.name
    trade_agent_params['exchange_name'] = exchange_name

    stg_run_info = StgRunInfo(stg_name=stg_class.__name__,  # '{.__name__}'.format(stg_class)
                              dt_from=datetime.now(),
                              # dt_to=None,
                              stg_params=json.dumps(strategy_params),
                              md_agent_params_list=json.dumps(md_agent_params_list_4_json),
                              run_mode=int(run_mode),
                              trade_agent_params_list=json.dumps(trade_agent_params_4_json))
    with with_db_session(engine_ibats) as session:
        session.add(stg_run_info)
        session.commit()
        stg_run_id = stg_run_info.stg_run_id
    # 初始化策略实体，传入参数
    stg_base = stg_class(**strategy_params)
    # 设置策略交易接口 trade_agent，这里不适用参数传递的方式而使用属性赋值，
    # 因为stg子类被继承后，参数主要用于设置策略所需各种参数使用
    stg_base.trade_agent = trader_agent_factory(run_mode, stg_run_id, **trade_agent_params)
    # 对不同周期设置相应的md_agent
    # 初始化各个周期的 md_agent
    md_period_agent_dic = {}
    for md_agent_param in md_agent_params_list:
        period = md_agent_param['md_period']
        md_agent = md_agent_factory(**md_agent_param)
        md_period_agent_dic[period] = md_agent
        # 对各个周期分别加载历史数据，设置对应 handler
        # 通过 md_agent 加载各个周期的历史数据,
        # 这里加载历史数据为初始化数据：
        # 主要对应于 md_agent_params_list 参数中 init_md_date_from init_md_date_to 等参数设置
        # 与下方的另一次加载历史数据不同，下面的加载历史数据位回测过程中对回测数据的加载，两者不可合并
        his_df_dic = md_agent.load_history()
        if his_df_dic is None:
            logger_stg_base.warning('加载 %s 历史数据为 None', period)
            continue
        if isinstance(his_df_dic, dict):
            md_df = his_df_dic['md_df']
        else:
            md_df = his_df_dic
            warnings.warn('load_history 返回 df 数据格式即将废弃，请更新成 dict', DeprecationWarning)

        context = {ContextKey.instrument_id_list: list(md_agent.instrument_id_set)}
        stg_base.load_md_period_df(period, md_df, context)
        logger_stg_base.debug('加载 %s 历史数据 %s 条', period, 'None' if md_df is None else str(md_df.shape[0]))

    # 初始化 StgHandlerBase 实例
    if run_mode == RunMode.Realtime:
        stg_handler = StgHandlerRealtime(stg_run_id=stg_run_id, stg_base=stg_base,
                                         md_period_agent_dic=md_period_agent_dic, **trade_agent_params)
    elif run_mode == RunMode.Backtest:
        stg_handler = StgHandlerBacktest(stg_run_id=stg_run_id, stg_base=stg_base,
                                         md_period_agent_dic=md_period_agent_dic, **trade_agent_params)
    else:
        raise ValueError('run_mode %d error' % run_mode)

    logger_stg_base.debug('初始化 %r 完成', stg_handler)
    return stg_handler
