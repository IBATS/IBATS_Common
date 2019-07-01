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
import time
import warnings
from abc import ABC
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from queue import Empty
from threading import Thread

from ibats_utils.db import with_db_session
from ibats_utils.mess import try_2_date, load_class, get_module_path
from sqlalchemy.exc import SQLAlchemyError

from ibats_common.backend import engines
from ibats_common.backend.orm import StgRunInfo, StgRunStatusDetail
from ibats_common.common import ExchangeName, RunMode, ContextKey, CalcMode
from ibats_common.config import config
from ibats_common.md import md_agent_factory
from ibats_common.strategy import StgBase
from ibats_common.trade import trader_agent_factory

engine_ibats = engines.engine_ibats
logger = logging.getLogger(__name__)


class StgHandlerBase(Thread, ABC):

    def __init__(self, stg_run_id, stg_base: StgBase, run_mode, md_key_period_agent_dic):
        super().__init__(daemon=True)
        self.stg_run_id = stg_run_id
        self.run_mode = run_mode
        # 初始化策略实体，传入参数
        self.stg_base = stg_base
        # 设置工作状态
        self.is_working = False
        self.is_done = False
        # 日志
        self.logger = logging.getLogger(str(self.__class__))
        # 对不同周期设置相应的md_agent
        self.md_key_period_agent_dic = md_key_period_agent_dic
        self.stg_run_status_detail_list = []

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
            try:
                session.bulk_save_objects(self.stg_run_status_detail_list)
                # sql_str = StgRunInfo.update().where(
                # StgRunInfo.c.stg_run_id == self.stg_run_id).values(dt_to=datetime.now())
                # session.execute(sql_str)
                session.commit()
                self.logger.debug("%d 条 stg_run_status_detail 被保存", len(self.stg_run_status_detail_list))
            except SQLAlchemyError:
                logger.exception("%d 条 stg_run_status_detail 被保存时发生异常", len(self.stg_run_status_detail_list))
                session.rollback()

        self.is_working = False
        self.is_done = True

    def __repr__(self):
        return '<{0.__class__.__name__}:{0.stg_run_id} {0.run_mode}>'.format(self)


class StgHandlerRealtime(StgHandlerBase):

    def __init__(self, stg_run_id, stg_base: StgBase, md_key_period_agent_dic, **kwargs):
        super().__init__(stg_run_id=stg_run_id, stg_base=stg_base, run_mode=RunMode.Realtime,
                         md_key_period_agent_dic=md_key_period_agent_dic)
        # 对不同周期设置相应的md_agent
        self.md_period_agent_dic = md_key_period_agent_dic
        # 设置线程池
        self.running_thread = {}
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

    def __init__(self, stg_run_id, stg_base: StgBase, run_mode, md_key_period_agent_dic, date_from, date_to,
                 md_td_agent_key_list_map=None, **kwargs):
        super().__init__(stg_run_id=stg_run_id, stg_base=stg_base, run_mode=run_mode,
                         md_key_period_agent_dic=md_key_period_agent_dic)
        # 设置回测时间区间
        self.date_from = try_2_date(date_from)
        self.date_to = try_2_date(date_to)
        self.params = kwargs
        if not isinstance(self.date_from, date):
            raise ValueError("date_from: %s", date_from)
        if not isinstance(self.date_to, date):
            raise ValueError("date_from: %s", date_to)
        # 用于维护 md_agent 与 td_agent 之间对应关系（多对多关系）
        # 用于在回测成交时指定行情发送到哪一个 trade_agent，默认情况下，该map将交易所相同的 md td agent进行匹配，也可以根据需要在参数设是手动进行配置
        self.logger.debug('设置 md_td_agent_key_list_map')
        if md_td_agent_key_list_map is not None:
            self.md_td_agent_key_list_map = md_td_agent_key_list_map
        else:
            self.md_td_agent_key_list_map = defaultdict(list)
            # 遍历所有 md_agent, 查找 exchange_name 相同的进行 mapping
            for md_agent_key, period_agent_dic in self.md_key_period_agent_dic.items():
                for period, md_agent in period_agent_dic.items():
                    for trade_agent_key, trade_agent in self.stg_base.trade_agent_dic.items():
                        if md_agent.exchange_name == trade_agent.exchange_name:
                            self.md_td_agent_key_list_map[md_agent_key].append(trade_agent_key)

                    break

        # 设置相应 trade_agent 的 关键 key 信息
        self.logger.debug('设置 关键 key 信息')
        for md_agent_key, trade_agent_key_list in self.md_td_agent_key_list_map.items():
            for period, md_agent in self.md_key_period_agent_dic[md_agent_key].items():
                for trade_agent_key in trade_agent_key_list:
                    md_agent.check_key()
                    self.stg_base.trade_agent_dic[trade_agent_key].set_timestamp_key(md_agent.timestamp_key)
                    self.stg_base.trade_agent_dic[trade_agent_key].set_symbol_key(md_agent.symbol_key)
                    self.stg_base.trade_agent_dic[trade_agent_key].set_close_key(md_agent.close_key)

                break

    def get_periods_history_iterator(self) -> dict:
        """
        获取各个周期的 历史数据的迭代器
        :return:
        """
        md_agent_key_cor_func_dic = {}
        # 每一个 md_agent 抓取第一条记录，放入 md_list_sorted_by_datetime_tag 并按 datetime_tag 排序
        for md_agent_key, period_agent_dic in self.md_key_period_agent_dic.items():
            for period, md_agent in period_agent_dic.items():
                cor_func = md_agent.cor_load_history_record(self.date_from, self.date_to, load_md_count=0)
                meta_dic = {
                    'symbol_key': md_agent.symbol_key,
                    'close_key': md_agent.close_key,
                    'timestamp_key': md_agent.timestamp_key
                }
                md_agent_key_cor_func_dic[(md_agent_key, period)] = (cor_func, meta_dic)
        return md_agent_key_cor_func_dic

    def load_history_record(self):
        """
        迭代器方法，用于产生行情数据
        :return:
        """
        md_list_sorted_by_datetime_tag = []
        # 每一个 md_agent 抓取第一条记录，放入 md_list_sorted_by_datetime_tag 并按 datetime_tag 排序
        for md_agent_key, period_agent_dic in self.md_key_period_agent_dic.items():
            for period, md_agent in period_agent_dic.items():
                cor_func = md_agent.cor_load_history_record(self.date_from, self.date_to, load_md_count=0)
                try:
                    num, datetime_tag, md_s = cor_func.send(None)
                except StopIteration as exp:
                    self.logger.info('%s %s 没有数据', md_agent_key, period)
                    continue

                md_list_sorted_by_datetime_tag.append((datetime_tag, md_agent_key, period, num, md_s, cor_func))

        # 开始循环，每一次循环：
        # 1）读取发送 datetime_tag 最小的一条记录
        # 2）从相应的 md_agent 再读取一条记录，补充到 md_list_sorted_by_datetime_tag
        # 3）按 datetime_tag 排序
        # 如此循环，知道全部数据被发送完
        md_list_sorted_by_datetime_tag.sort(key=lambda x: x[0])
        list_count = len(md_list_sorted_by_datetime_tag)
        data_count = 0
        while list_count > 0:
            datetime_tag, md_agent_key, period, num, md_s, cor_func = md_list_sorted_by_datetime_tag[0]
            yield datetime_tag, md_agent_key, period, num, md_s
            try:
                num, datetime_tag, md_s = cor_func.send(None)
                md_list_sorted_by_datetime_tag[0] = (datetime_tag, md_agent_key, period, num, md_s, cor_func)
            except StopIteration as exp:
                data_count += exp.value
                self.logger.info('%s %s 推送 %d 条数据完成', md_agent_key, period, exp.value)
                md_list_sorted_by_datetime_tag.pop(0)
                list_count = len(md_list_sorted_by_datetime_tag)

            # 重新按 datetime_tag 排序
            md_list_sorted_by_datetime_tag.sort(key=lambda x: x[0])

        self.logger.info('全部数据推送完成， 累计推送 %d 条数据', data_count)

    def _update_stg_run_status_detail(self, trade_agent_status_detail_list):
        if trade_agent_status_detail_list is not None and len(trade_agent_status_detail_list) > 0:
            if len(self.stg_run_status_detail_list) == 0:
                detail_last = StgRunStatusDetail.create_t_1(
                    self.stg_run_id, trade_agent_status_detail_list)
                self.stg_run_status_detail_list.append(detail_last)
            else:
                detail_last = self.stg_run_status_detail_list[-1]

            detail = detail_last.update_by_trade_agent_status_detail_list(
                trade_agent_status_detail_list)
            self.stg_run_status_detail_list.append(detail)

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
            # 对每个 md_agent 激活 load_history_record
            # 每一次取记录包括：时间轴标签 + 行情数据
            # 比较各个 md_agent 取时间最小的作为本次数据推送，如此循环，直至所有 mg_agent 记录均提取完成
            # 按照时间顺序将各个周期数据依次推入对应 handler
            data_count = 0
            datetime_tag_last = None
            for data_count, (datetime_tag, md_agent_key, period, num, md_s) in enumerate(
                    self.load_history_record(), start=1):
                md = md_s.to_dict()
                # self.logger.debug("md: %s", md)
                # 在回测阶段，需要对 trade_agent 设置最新的md数据，一遍交易接口确认相应的k线日期
                for trade_agent_key in self.md_td_agent_key_list_map[md_agent_key]:
                    if trade_agent_key == ExchangeName.Default:
                        continue
                    self.stg_base.trade_agent_dic[trade_agent_key].set_curr_md(period, md)

                # 执行策略相应的事件响应函数
                self.stg_base.on_period_md_handler(period, md, md_agent_key)
                # 根据最新的 md 及 持仓信息 更新 账户信息
                for trade_agent_key in self.md_td_agent_key_list_map[md_agent_key]:
                    if trade_agent_key == ExchangeName.Default:
                        continue
                    self.stg_base.trade_agent_dic[trade_agent_key].update_trade_agent_status_detail()

                # 以下两种情况开始计算 StgRunStatusDetail：
                # 1）config.UPDATE_STG_RUN_STATUS_DETAIL_PERIOD == 1 时，每一次状态更新均重新计算
                # 2）每一个自然日开始
                # 首次运行的时候，回触发一次初始化，先生成一条 T-1日的状态记录
                if config.UPDATE_STG_RUN_STATUS_DETAIL_PERIOD == 1:
                    trade_agent_status_detail_list = [
                        trade_agent.trade_agent_status_detail_latest
                        for key, trade_agent in self.stg_base.trade_agent_dic.items()
                        if key != ExchangeName.Default and trade_agent.trade_agent_status_detail_latest is not None]

                elif (config.UPDATE_STG_RUN_STATUS_DETAIL_PERIOD == 2
                      and datetime_tag_last is not None
                      and datetime_tag_last.date() != datetime_tag.date()):
                    # TODO: 这里应该是找前一交易日的最后一个 trade_agent_status_detail
                    trade_agent_status_detail_list = [
                        trade_agent.trade_agent_status_detail_latest
                        for key, trade_agent in self.stg_base.trade_agent_dic.items()
                        if key != ExchangeName.Default and trade_agent.trade_agent_status_detail_latest is not None]
                else:
                    trade_agent_status_detail_list = None

                self._update_stg_run_status_detail(trade_agent_status_detail_list)

                datetime_tag_last = datetime_tag

            self.logger.info('执行回测任务【%s - %s】完成，处理数据 %d 条', self.date_from, self.date_to, data_count)
        finally:
            if config.UPDATE_STG_RUN_STATUS_DETAIL_PERIOD == 2:
                trade_agent_status_detail_list = [
                    trade_agent.trade_agent_status_detail_latest
                    for trade_agent in self.stg_base.trade_agent_dic.values()
                    if trade_agent is not None]
                self._update_stg_run_status_detail(trade_agent_status_detail_list)

            self.stg_run_ending()
            self.is_working = False


def strategy_handler_factory(
        stg_class: type(StgBase), strategy_params, md_agent_params_list, run_mode: RunMode, exchange_name: ExchangeName,
        trade_agent_params: dict, strategy_handler_param: dict) -> StgHandlerBase:
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
    :param strategy_handler_param: strategy_handler 运行参数
    :return: 策略执行对象实力
    """
    # 为 md_agent_param 补充参数
    for md_agent_param in md_agent_params_list:
        md_agent_param['exchange_name'] = exchange_name

    # 为 trade_agent_params 补充参数
    trade_agent_params['exchange_name'] = exchange_name
    trade_agent_params['is_default'] = True
    trade_agent_params_list = [trade_agent_params]
    stg_handler = strategy_handler_factory_multi_exchange(
        stg_class, strategy_params, md_agent_params_list, run_mode, trade_agent_params_list, strategy_handler_param)
    return stg_handler


def strategy_handler_factory_multi_exchange(
        stg_class: type(StgBase), strategy_params, md_agent_params_list, run_mode: RunMode,
        trade_agent_params_list: list, strategy_handler_param: dict, add_2_db=True, stg_run_id=None,
        is_4_shown=False) -> StgHandlerBase:
    """
    多交易所策略处理具备
    建立策略对象
    建立数据库相应记录信息
    根据运行模式（实时、回测）：选择相应的md_agent以及trade_agent
    :param stg_class: 策略类型 StgBase 的子类
    :param strategy_params: 策略参数
    :param md_agent_params_list: 行情代理（md_agent）参数，支持同时订阅多周期、多品种，
    例如：同时订阅 [ethusdt, eosusdt] 1min 行情、[btcusdt, ethbtc] tick 行情
    :param run_mode: 运行模式 RunMode.Realtime  或 RunMode.Backtest
    :param trade_agent_params_list: 运行参数，回测模式下：运行起止时间，实时行情下：加载定时器等设置
    :param strategy_handler_param: strategy_handler 运行参数
    :param add_2_db: 默认情况下，自动将 stg_run_info 添加到数据库
    :param stg_run_id: 仅在 add_2_db == False 时需要添加 stg_run_id
    :param is_4_shown: 仅供回溯展示需要，不加载 strategy_class 实例及其相关方法
    :return: 策略执行对象实力
    """
    stg_module = get_module_path(stg_class)
    stg_run_info = StgRunInfo(stg_name=stg_class.__name__,  # 例如："MACrossStg"
                              stg_module=stg_module,  # 例如："ibats_common.example.ma_cross_stg"
                              dt_from=datetime.now(),
                              # dt_to=None,
                              stg_params=json.dumps(strategy_params),
                              md_agent_params_list=json.dumps(md_agent_params_list),
                              run_mode=int(run_mode),
                              trade_agent_params_list=json.dumps(trade_agent_params_list),
                              strategy_handler_param=json.dumps(strategy_handler_param),
                              )
    if add_2_db:
        with with_db_session(engine_ibats) as session:
            session.add(stg_run_info)
            session.commit()
            stg_run_id = stg_run_info.stg_run_id

    # 初始化策略实体，传入参数
    if is_4_shown:
        stg_base = StgBase(**strategy_params)
    else:
        stg_base = stg_class(**strategy_params)
    # 设置相关属性
    stg_base.stg_run_id = stg_run_id
    logger.debug('stg_run_id=%d, strategy_params: %s', stg_run_id, strategy_params)
    # 设置策略交易接口 trade_agent，这里不适用参数传递的方式而使用属性赋值，
    # 因为stg_base子类被继承后，参数主要用于设置策略所需各种参数使用
    for num, params in enumerate(trade_agent_params_list, start=1):
        logger.debug('%d) run_mode=%s, stg_run_id=%d, trade_agent_params: %s', num, run_mode.name, stg_run_id, params)
        if run_mode == RunMode.Backtest_FixPercent and params['calc_mode'] == CalcMode.Margin:
            raise ValueError(f'{num}) {RunMode.Backtest_FixPercent.name} 模式下，trade_agent 不支持 CalcMode.Margin 模式，'
                             f'请切换到 CalcMode.Normal 模式')
        # 默认使用交易所名称，若同一交易所，多个账户交易，则可以单独指定名称
        agent_name = params.setdefault('agent_name', params['exchange_name'])
        trade_agent = trader_agent_factory(run_mode, stg_run_id, **params)
        stg_base.trade_agent_dic[agent_name] = trade_agent
        if 'is_default' in params and params['is_default']:
            stg_base.trade_agent = trade_agent
            stg_base.trade_agent_dic[ExchangeName.Default] = trade_agent

    # 对不同周期设置相应的md_agent
    # 初始化各个周期的 md_agent
    md_key_period_agent_dic = defaultdict(dict)
    for num, params in enumerate(md_agent_params_list, start=1):
        period = params['md_period']
        agent_name = params.setdefault('agent_name', f"{params['exchange_name']}.{period}")
        logger.debug('%d) run_mode=%s, md_agent_params: %s', num, run_mode.name, params)
        md_agent = md_agent_factory(run_mode=run_mode, **params)
        md_key_period_agent_dic[agent_name][period] = md_agent
        # 对各个周期分别加载历史数据，设置对应 handler
        # 通过 md_agent 加载各个周期的历史数据,
        # 这里加载历史数据为初始化数据：
        # 主要对应于 md_agent_params_list 参数中 init_md_date_from init_md_date_to 等参数设置
        # 与下方的另一次加载历史数据不同，下面的加载历史数据位回测过程中对回测数据的加载，两者不可合并
        his_df_dic = md_agent.load_history()
        if his_df_dic is None:
            logger.warning('加载 %s 历史数据为 None', period)
            continue
        if isinstance(his_df_dic, dict):
            md_df = his_df_dic['md_df']
        else:
            md_df = his_df_dic
            warnings.warn('load_history 返回 df 数据格式即将废弃，请更新成 dict', DeprecationWarning)

        context = {
            ContextKey.instrument_id_list: list(md_agent.instrument_id_list),
            ContextKey.md_agent_key: agent_name,
            ContextKey.period: period,
        }
        stg_base.load_md_period_df(period, md_df, context)
        logger.debug('加载 %s 历史数据 %s 条', period, 'None' if md_df is None else str(md_df.shape[0]))

    # 设置 md_td_agent_key_list_map 用于标识 md_agent_key 与 trade_agent_key 之间的对应关系
    # md_agent_key 与 trade_agent_key 是 1对多 关系
    if 'md_td_agent_key_list_map' in strategy_handler_param:
        md_td_agent_key_list_map = strategy_handler_param['md_td_agent_key_list_map']
    else:
        md_td_agent_key_list_map = defaultdict(list)
        # 遍历所有 md_agent, 查找 exchange_name 相同的进行 mapping
        for md_agent_key, period_agent_dic in md_key_period_agent_dic.items():
            for period, md_agent in period_agent_dic.items():
                for trade_agent_key, trade_agent in stg_base.trade_agent_dic.items():
                    if md_agent.exchange_name == trade_agent.exchange_name:
                        md_td_agent_key_list_map[md_agent_key].append(trade_agent_key)

                break
        # 整理 mapping 去除重复 key，整理过程中 td_agent_key_list 顺序可能被打乱
        for md_agent_key in list(md_td_agent_key_list_map.keys()):
            td_agent_key_list = list(set(md_td_agent_key_list_map[md_agent_key]))
            md_td_agent_key_list_map[md_agent_key] = td_agent_key_list
        strategy_handler_param['md_td_agent_key_list_map'] = md_td_agent_key_list_map
    # TODO: 可以增加一个合法性检查，检查是否 md_agent_key 与 trade_agent_key 是 1对多 关系
    logger.debug('md_td_agent_key_list_map: %s', md_td_agent_key_list_map)

    # 为 stg_base 设置 md_td_agent_key_list_map
    stg_base.set_md_td_agent_key_list_map(md_td_agent_key_list_map)

    # 初始化 StgHandlerBase 实例
    logger.debug("stg_run_id=%d,\n\tstrategy_handler_param: %s", stg_run_id, strategy_handler_param)
    if run_mode == RunMode.Realtime:
        stg_handler = StgHandlerRealtime(
            stg_run_id=stg_run_id, stg_base=stg_base, md_key_period_agent_dic=md_key_period_agent_dic,
            **strategy_handler_param)
    elif run_mode in (RunMode.Backtest, RunMode.Backtest_FixPercent):
        stg_handler = StgHandlerBacktest(
            stg_run_id=stg_run_id, stg_base=stg_base, run_mode=run_mode,
            md_key_period_agent_dic=md_key_period_agent_dic,
            **strategy_handler_param)
    else:
        raise ValueError('run_mode %d error' % run_mode)

    logger.debug('初始化 %r 完成', stg_handler)
    return stg_handler


@lru_cache()
def strategy_handler_loader(stg_run_id, is_4_shown=True) -> StgHandlerBase:
    """
    根据 stg_run_id 从数据库中加载相应参数，动态加载策略类，实例化 stg_handler 类
    :param stg_run_id:
    :param is_4_shown: 仅供回溯展示需要，不加载 strategy_class 实例及其相关方法
    :return:
    """
    with with_db_session(engine_ibats) as session:
        stg_run_info = session.query(StgRunInfo).filter(StgRunInfo.stg_run_id == stg_run_id).first()
    if stg_run_info is None:
        raise KeyError(f'stg_run_id={stg_run_id} 无效，数据库中没有相关数据')
    # 2019-05-14 废弃参数 module_name_replacement_if_main
    # 因版本升级该问题以及被解决，因此该参数已经无用
    # if module_name_replacement_if_main is not None and stg_run_info.stg_module == '__main__':
    #     # stg_run_info.stg_module = "ibats_common.example.ma_cross_stg"
    #     stg_run_info.stg_module = module_name_replacement_if_main

    stg_class = load_class(stg_run_info.stg_module, stg_run_info.stg_name)

    strategy_params = json.loads(stg_run_info.stg_params)
    md_agent_params_list = json.loads(stg_run_info.md_agent_params_list)
    run_mode = RunMode(stg_run_info.run_mode)
    trade_agent_params_list = json.loads(stg_run_info.trade_agent_params_list)
    strategy_handler_param = json.loads(stg_run_info.strategy_handler_param)
    stg_handler = strategy_handler_factory_multi_exchange(
        stg_class=stg_class,
        strategy_params=strategy_params,
        md_agent_params_list=md_agent_params_list,
        run_mode=run_mode,
        trade_agent_params_list=trade_agent_params_list,
        strategy_handler_param=strategy_handler_param,
        add_2_db=False,
        stg_run_id=stg_run_info.stg_run_id,
        is_4_shown=is_4_shown,
    )
    return stg_handler
