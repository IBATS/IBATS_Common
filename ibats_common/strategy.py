# -*- coding: utf-8 -*-
"""
Created on 2017/9/2
@author: MG
@contact : mmmaaaggg@163.com
@desc    : 策略基类，所有策略均集成此基类
"""
import logging
from collections import defaultdict

import pandas as pd

from ibats_common.common import PeriodType, ExchangeName, ContextKey, Direction

logger_stg_base = logging.getLogger(__name__)


class StgBase:

    def __init__(self, *args, **kwargs):
        self.stg_run_id = None
        # 记录各个md_agent_key、各个周期 md 数据
        self._md_agent_key_period_df_dic = defaultdict(dict)
        # 记录各个md_agent_key、各个周期 md 列信息
        self._md_agent_key_period_df_col_name_list_dic = defaultdict(dict)
        # 记录各个md_agent_key、各个周期 context 信息
        self._md_agent_key_period_context_dic = defaultdict(dict)
        # 记录各个md_agent_key对应的 td_agent_key list
        self._md_td_agent_key_list_map = defaultdict(list)
        self._td_md_agent_key_map = {}
        # 记录在行情推送过程中最新的一笔md数据
        # self._period_curr_md_dic = {}
        self.trade_agent = None
        self.trade_agent_dic = {}
        self.logger = logging.getLogger(str(self.__class__))
        self._on_period_event_dic = {
            PeriodType.Tick: EventHandlersRelation(
                PeriodType.Tick, self.on_prepare_tick, self.on_tick, self.on_tick_release, pd.DataFrame),
            PeriodType.Min1: EventHandlersRelation(
                PeriodType.Min1, self.on_prepare_min1, self.on_min1, self.on_min1_release, pd.DataFrame),
            PeriodType.Hour1: EventHandlersRelation(
                PeriodType.Hour1, self.on_prepare_hour1, self.on_hour1, self.on_hour1_release, pd.DataFrame),
            PeriodType.Day1: EventHandlersRelation(
                PeriodType.Day1, self.on_prepare_day1, self.on_day1, self.on_day1_release, pd.DataFrame),
            PeriodType.Week1: EventHandlersRelation(
                PeriodType.Week1, self.on_prepare_week1, self.on_week1, self.on_week1_release, pd.DataFrame),
            PeriodType.Mon1: EventHandlersRelation(
                PeriodType.Mon1, self.on_prepare_month1, self.on_month1, self.on_month1_release, pd.DataFrame),
        }
        self.args = args
        self.kwargs = kwargs

    def set_md_td_agent_key_list_map(self, md_td_agent_key_list_map):
        self._md_td_agent_key_list_map = md_td_agent_key_list_map
        # 反过来 td_md_agent_key_list_map 建立 trade_agent_key 与 md_agent_key 之间的对应关系
        self.logger.debug('trade_agent_key 与 md_agent_key 对应关系')
        num = 0
        for md_agent_key, td_agent_key_list in md_td_agent_key_list_map.items():
            for num, td_agent_key in enumerate(td_agent_key_list, start=num + 1):
                self._td_md_agent_key_map[td_agent_key] = md_agent_key
                if isinstance(td_agent_key, ExchangeName):
                    self._td_md_agent_key_map[str(td_agent_key)] = md_agent_key
                    self._td_md_agent_key_map[td_agent_key.value] = md_agent_key
                if isinstance(td_agent_key, int):
                    self._td_md_agent_key_map[ExchangeName(td_agent_key)] = md_agent_key
                    self._td_md_agent_key_map[ExchangeName(td_agent_key).name] = md_agent_key
                self.logger.debug("%d) %s -> %s", num, td_agent_key, md_agent_key)

    def on_timer(self):
        pass

    def load_md_period_df(self, period, md_df: pd.DataFrame, context):
        """初始化加载 md 数据"""
        md_agent_key = context[ContextKey.md_agent_key]
        self._md_agent_key_period_df_dic[md_agent_key][period] = md_df
        self._md_agent_key_period_df_col_name_list_dic[md_agent_key][period] = list(md_df.columns) if isinstance(
            md_df, pd.DataFrame) else None
        self._md_agent_key_period_context_dic[md_agent_key][period] = context
        # prepare_event_handler = self._on_period_prepare_event_dic[period]
        prepare_event_handler = self._on_period_event_dic[period].prepare_event
        prepare_event_handler(md_df, context)

    def init(self):
        """
        加载历史数据后，启动周期策略执行函数之前
        执行初始化动作，连接 trade_agent
        以后还可以放出其他初始化动作
        :return:
        """
        # self.trade_agent.connect()
        for num, (name, trade_agent) in enumerate(self.trade_agent_dic.items()):
            # ExchangeName.Default 作为默认 trade_agent 在 trade_agent_dic 中存在重复实例，
            # 因此无需对该 key 进行操作
            if name == ExchangeName.Default:
                continue
            self.logger.debug('%d) init trade_agent %s', num, name)
            trade_agent.connect()

    def release(self):
        """
        释放资源
        :return:
        """
        # 行情信息处理函数调用结束
        # 调用各个 md_agent_key 的各个 period 的 汇总函数
        # self._md_agent_key_period_df_dic[md_agent_key][period]
        for md_agent_key, _ in self._md_agent_key_period_df_dic.items():
            for period, md_df in _.items():
                event_handler = self._on_period_event_dic[period].md_release_event
                try:
                    event_handler(md_df)
                except:
                    self.logger.exception('period=%s %s invoked exception', period, event_handler)

        for num, (name, trade_agent) in enumerate(self.trade_agent_dic.items()):
            # ExchangeName.Default 作为默认 trade_agent 在 trade_agent_dic 中存在重复实例，
            # 因此无需对该 key 进行操作
            if name == ExchangeName.Default:
                continue
            self.logger.debug('%d) release trade_agent %s', num, name)
            trade_agent.release()

    def on_period_md_handler(self, period, md, md_agent_key):
        """响应 period 数据"""
        # 本机测试，延时0.155秒，从分钟K线合成到交易策略端收到数据
        period_event_relation = self._on_period_event_dic[period]
        event_handler = period_event_relation.md_event
        param_type = period_event_relation.param_type
        context = self._md_agent_key_period_context_dic[md_agent_key][period]
        # TODO 由于每一次进入都需要进行判断，增加不必要的计算，考虑通过优化提高运行效率
        if param_type is dict:
            param = md
        elif param_type is pd.DataFrame:
            # 2019-06-05
            # 移除 self._on_period_md_append 该函数调用，将函数内部代码复制到下面直接使用
            # param = self._on_period_md_append(period, md, md_agent_key)

            md_df = pd.DataFrame([md])
            # self.logger.debug('%s -> %s', period, md)
            if period in self._md_agent_key_period_df_dic[md_agent_key]:
                col_name_list = self._md_agent_key_period_df_col_name_list_dic[md_agent_key][period]
                md_df_his = self._md_agent_key_period_df_dic[md_agent_key][period].append(md_df[col_name_list])
                self._md_agent_key_period_df_dic[md_agent_key][period] = md_df_his
            else:
                md_df_his = md_df
                self._md_agent_key_period_df_dic[md_agent_key][period] = md_df_his
                self._md_agent_key_period_df_col_name_list_dic[md_agent_key][period] = \
                    list(md_df.columns) if isinstance(md_df, pd.DataFrame) else None
            param = md_df_his
        else:
            raise ValueError("不支持 %s 类型作为 %s 的事件参数" % (param_type, period))
        event_handler(param, context)

    def on_prepare_tick(self, md_df, context):
        """Tick 历史数据加载执行语句"""
        pass

    def on_prepare_min1(self, md_df, context):
        """1分钟线 历史数据加载执行语句"""
        pass

    def on_prepare_hour1(self, md_df, context):
        """1小时线 历史数据加载执行语句"""
        pass

    def on_prepare_day1(self, md_df, context):
        """1日线 历史数据加载执行语句"""
        pass

    def on_prepare_week1(self, md_df, context):
        """1周线 历史数据加载执行语句"""
        pass

    def on_prepare_month1(self, md_df, context):
        """1月线 历史数据加载执行语句"""
        pass

    def on_tick(self, md_df, context):
        """Tick策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_min1(self, md_df, context):
        """1分钟线策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_hour1(self, md_df, context):
        """1小时线策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_day1(self, md_df, context):
        """1日线策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_week1(self, md_df, context):
        """1周线策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_month1(self, md_df, context):
        """1月线策略执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_tick_release(self, md_df):
        """Tick行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_min1_release(self, md_df):
        """1分钟线行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_hour1_release(self, md_df):
        """1小时线行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_day1_release(self, md_df):
        """1日线行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_week1_release(self, md_df):
        """1周线行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def on_month1_release(self, md_df):
        """1月线行情结束执行语句，需要相应策略实现具体的策略算法"""
        pass

    def open_long(self, instrument_id, price, vol, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            trade_agent = self.trade_agent
            # self.logger.info("  %s %s  open  long  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug("  %s %s ↗   %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.open_long(instrument_id, price, vol)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)

            trade_agent = self.trade_agent_dic[trade_agent_key]
            # self.logger.info("  %s %s  open  long  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug("  %s %s ↗   %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.open_long(instrument_id, price, vol)

    def close_long(self, instrument_id, price, vol, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            trade_agent = self.trade_agent
            # self.logger.info(" %s %s close  long  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug(" %s %s ↘   %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.close_long(instrument_id, price, vol)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)

            trade_agent = self.trade_agent_dic[trade_agent_key]
            # self.logger.info(" %s %s close  long  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug(" %s %s ↘   %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.close_long(instrument_id, price, vol)

    def open_short(self, instrument_id, price, vol, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            trade_agent = self.trade_agent
            # self.logger.info(" %s %s  open short  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug(" %s %s   ↘ %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.open_short(instrument_id, price, vol)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)

            trade_agent = self.trade_agent_dic[trade_agent_key]
            # self.logger.info(" %s %s  open short  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug(" %s %s   ↘ %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.open_short(instrument_id, price, vol)

    def close_short(self, instrument_id, price, vol, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            trade_agent = self.trade_agent
            # self.logger.info("%s %s close short  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug("%s %s   ↗ %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.close_short(instrument_id, price, vol)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)

            trade_agent = self.trade_agent_dic[trade_agent_key]
            # self.logger.info("%s %s close short  %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            self.logger.debug("%s %s   ↗ %.2f * %f", trade_agent.curr_timestamp, instrument_id, price, vol)
            return trade_agent.close_short(instrument_id, price, vol)

    def get_position(self, instrument_id, trade_agent_key=None, md_agent_key=None, **kwargs) -> dict:
        """
        position_date 作为key， PosStatusInfo 为 val
        返回 position_date_pos_info_dic
        :param instrument_id:
        :param trade_agent_key: trade_agent key值，默认为None，选择默认 trade_agent
        :return:
        """
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.get_position(instrument_id, **kwargs)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].get_position(instrument_id, **kwargs)

    def get_order(self, instrument_id, trade_agent_key=None, md_agent_key=None) -> list:
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.get_order(instrument_id)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].get_order(instrument_id)

    def cancel_order(self, instrument_id, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.cancel_order(instrument_id)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].cancel_order(instrument_id)

    @property
    def datetime_last_update_position(self, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.datetime_last_update_position
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].datetime_last_update_position

    @property
    def datetime_last_rtn_trade_dic(self, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.datetime_last_rtn_trade_dic
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].datetime_last_rtn_trade_dic

    @property
    def datetime_last_update_position_dic(self, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.datetime_last_update_position_dic
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].datetime_last_update_position_dic

    @property
    def datetime_last_send_order_dic(self, trade_agent_key=None, md_agent_key=None):
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.datetime_last_send_order_dic
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].datetime_last_send_order_dic

    def get_balance(self, non_zero_only=True, trade_type_only=True, currency=None, force_refresh=False,
                    trade_agent_key=None, md_agent_key=None) -> dict:
        """
        调用接口 查询 各个币种仓位
        :param non_zero_only: 只保留非零币种
        :param trade_type_only: 只保留 trade 类型币种，frozen 类型的不保存
        :param currency: 只返回制定币种 usdt eth 等
        :param force_refresh: 强制刷新，默认没30秒允许重新查询一次
        :param trade_agent_key: trade_agent key值，默认为None，选择默认 trade_agent
        :return: {'usdt': {<PositionDateType.History: 2>: {'currency': 'usdt', 'type': 'trade', 'balance': 144.09238}}}
        """
        if md_agent_key is None and trade_agent_key is None:
            return self.trade_agent.get_balance(non_zero_only, trade_type_only, currency, force_refresh)
        else:
            if trade_agent_key is None:
                trade_agent_key = self.get_td_agent_key(md_agent_key)
            return self.trade_agent_dic[trade_agent_key].get_balance(
                non_zero_only, trade_type_only, currency, force_refresh)

    def get_holding_currency(self, force_refresh=False, exclude_usdt=True,
                             trade_agent_key=None, md_agent_key=None) -> dict:
        """
        持仓情况dict（非usdt）,仅包含交易状态 type = 'trade' 的记录
        :param force_refresh:
        :param exclude_usdt: 默认为True，剔除 usdt
        :param trade_agent_key: trade_agent key值，默认为None，选择默认 trade_agent
        :param md_agent_key: md_agent_key key值，默认为None，指明与 trade_agent_key 对应的 md_agent_key
        :return:
         {'eos': {<PositionDateType.History: 2>: {'currency': 'eos', 'type': 'trade', 'balance': 144.09238}}}
        """
        cur_balance_dic = self.get_balance(non_zero_only=True, force_refresh=force_refresh,
                                           trade_agent_key=trade_agent_key, md_agent_key=md_agent_key)

        balance_dic = {}
        for currency, dic in cur_balance_dic.items():
            if exclude_usdt and currency == 'usdt':
                continue
            for pos_date_type, dic_sub in dic.items():
                if dic_sub['type'] != 'trade':
                    continue
                balance_dic.setdefault(currency, {})[pos_date_type] = dic_sub
        return balance_dic

    def get_td_agent_keys(self, md_agent_key):
        """返回首个 md_agent_key 对应的 td_agent_key 列表"""
        return self._md_td_agent_key_list_map[md_agent_key]

    def get_td_agent_key(self, md_agent_key):
        """返回首个 md_agent_key 对应的 td_agent_key"""
        key_list = self.get_td_agent_keys(md_agent_key)
        if len(key_list) == 0:
            return None
        else:
            return key_list[0]

    def keep_long(self, instrument_id, price, position):
        """
        保持多头持仓，如果没有仓位则买入，如果有仓位则继续持有
        :return:
        """
        position_date_pos_info_dic = self.get_position(instrument_id)
        no_target_position = True
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Short:
                    self.close_short(instrument_id, price, pos_info.position)
                elif direction == Direction.Long:
                    no_target_position = False
        if no_target_position:
            self.open_long(instrument_id, price, position)
        else:
            self.logger.debug("  %s %s *   %.2f holding", self.trade_agent.curr_timestamp, instrument_id, price)

    def keep_short(self, instrument_id, price, position):
        """
        保持空头持仓，如果没有仓位则卖出，如果有仓位则继续持有
        :return:
        """
        position_date_pos_info_dic = self.get_position(instrument_id)
        no_holding_target_position = True
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Long:
                    self.close_long(instrument_id, price, pos_info.position)
                elif direction == Direction.Short:
                    no_holding_target_position = False
        if no_holding_target_position:
            self.open_short(instrument_id, price, position)
        else:
            self.logger.debug(" %s %s   * %.2f holding", self.trade_agent.curr_timestamp, instrument_id, price)

    def keep_empty(self, instrument_id, price):
        """
        保持空头持仓，如果没有仓位则卖出，如果有仓位则继续持有
        :return:
        """
        position_date_pos_info_dic = self.get_position(instrument_id)
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Long:
                    self.close_long(instrument_id, price, pos_info.position)
                elif direction == Direction.Short:
                    self.close_short(instrument_id, price, pos_info.position)


class EventHandlersRelation:
    """
    用于记录事件类型与其对应的各种相关事件句柄之间的关系
    """

    def __init__(self, period_type, prepare_event, md_event, md_release_event, param_type):
        self.period_type = period_type
        self.prepare_event = prepare_event
        self.md_event = md_event
        self.md_release_event = md_release_event
        self.param_type = param_type
