#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from collections import defaultdict
from sqlalchemy.sql import func
from ibats_utils.db import with_db_session, get_db_session, execute_scalar, get_db_session
from ibats_common.backend import engines
import pandas as pd
from ibats_common.backend.orm import StgRunStatusDetail, OrderDetail, TradeDetail, StgRunInfo
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
from ibats_common.common import Action, Direction
from ibats_common.strategy_handler import stategy_handler_loader

logger = logging.getLogger(__name__)


def show_cash_and_margin(stg_run_id):
    """
    plot cash_and_margin
    :param stg_run_id:
    :return:
    """
    # stg_run_id=154
    engine_ibats = engines.engine_ibats
    with with_db_session(engine_ibats) as session:
        if stg_run_id is None:
            logger.warning('没有设置 stg_run_id 参数，将输出最新的 stg_run_id 对应记录')
            stg_run_id = session.query(func.max(StgRunInfo.stg_run_id)).scalar()

        sql_str = str(
            session.query(
                StgRunStatusDetail.trade_dt.label('trade_dt'),
                StgRunStatusDetail.cash_and_margin.label('cash_and_margin'),
            ).filter(
                StgRunStatusDetail.stg_run_id == stg_run_id
            )
        )

    df = pd.read_sql(sql_str, engine_ibats, params=[stg_run_id], index_col=['trade_dt'])
    df.plot()
    plt.show()


def show_order(stg_run_id, module_name_replacement_if_main='ibats_common.example.ma_cross_stg') -> defaultdict(
    lambda: defaultdict(list)):
    """
    plot candle and buy and sell point
    :param stg_run_id:
    :param module_name_replacement_if_main:
    :return:
    """
    # 加载数据库 engine
    engine_ibats = engines.engine_ibats
    # stg_run_id=1
    if stg_run_id is None:
        logger.warning('没有设置 stg_run_id 参数，将输出最新的 stg_run_id 对应记录')
        with with_db_session(engine_ibats) as session:
            stg_run_id = session.query(func.max(StgRunInfo.stg_run_id)).scalar()

    stg_handler = stategy_handler_loader(
        stg_run_id, module_name_replacement_if_main=module_name_replacement_if_main, is_4_shown=True)
    # 获取全部订单
    # session = get_db_session(engine_ibats)
    with with_db_session(engine_ibats) as session:
        order_detail_list_tot = session.query(
            OrderDetail
            # OrderDetail.order_dt.label('order_dt'),
            # OrderDetail.order_price.label('order_price'),
            # OrderDetail.action.label('action'),
            # OrderDetail.direction.label('direction'),
            # OrderDetail.order_vol.label('order_vol'),
            # OrderDetail.symbol.label('symbol'),
            # OrderDetail.trade_agent_key.label('trade_agent_key'),
        ).filter(
            OrderDetail.stg_run_id == stg_run_id
        ).all()
    # 根据 md_agent 进行分组
    md_agent_key_order_detail_list_dic = defaultdict(list)
    for num, order_detail in enumerate(order_detail_list_tot):
        md_agent_key = stg_handler.stg_base._td_md_agent_key_map[order_detail.trade_agent_key]
        md_agent_key_order_detail_list_dic[md_agent_key].append(order_detail)

    # 获取历史行情数据
    md_agent_key_cor_func_dic = stg_handler.get_periods_history_iterator()
    agent_count = len(md_agent_key_cor_func_dic)
    data_dict = defaultdict(lambda: defaultdict(list))
    # 根据 md_agent 对每一组行情 以及 对应的 order_detail_list 进行 plot
    # fig = plt.figure(1, figsize=(20, 4.8 * agent_count))
    for num, ((md_agent_key, period), (cor_func, meta_dic)) in enumerate(md_agent_key_cor_func_dic.items(), start=1):
        order_detail_list = md_agent_key_order_detail_list_dic[md_agent_key]
        df = pd.DataFrame([md_s for num, datetime_tag, md_s in cor_func])
        if df.shape[0] == 0:
            continue
        # ax = fig.add_subplot(num, 1, 1)
        # 行情
        symbol_key = meta_dic['symbol_key']
        close_key = meta_dic['close_key']
        timestamp_key = meta_dic['timestamp_key']
        for symbol, df_by_symbol in df.groupby(symbol_key):
            # df_by_symbol.set_index(timestamp_key)[close_key].plot(ax=ax, colormap='jet')
            data_dict[(md_agent_key, period, symbol)]['md'].append(df_by_symbol.set_index(timestamp_key)[close_key])
            # 开仓
            order_detail_list_sub = [_ for _ in order_detail_list
                                     if _.symbol == symbol
                                     and ((_.direction == Direction.Long.value and _.action == Action.Open.value)
                                          or (_.direction == Direction.Short.value and _.action != Action.Open.value)
                                          )
                                     ]
            trade_date_list = [_.order_dt for _ in order_detail_list_sub]
            price = [_.order_price for _ in order_detail_list_sub]
            # ax.scatter(trade_date_list, price, c='r', marker='^')
            data_dict[(md_agent_key, period, symbol)]['long_open_or_short_close'].append((trade_date_list, price))
            # 关仓
            order_detail_list_sub = [_ for _ in order_detail_list
                                     if (_.direction == Direction.Long.value and _.action != Action.Open.value)
                                     or (_.direction == Direction.Short.value and _.action == Action.Open.value)]
            trade_date_list = [_.order_dt for _ in order_detail_list_sub]
            price = [_.order_price for _ in order_detail_list_sub]
            # ax.scatter(trade_date_list, price, c='g', marker='v')
            data_dict[(md_agent_key, period, symbol)]['short_open_or_long_close'].append((trade_date_list, price))
            # 建立连线
            order_detail_list_symbol = [_ for _ in order_detail_list if _.symbol == symbol]
            for point1, point2 in zip(order_detail_list_symbol[:-1], order_detail_list_symbol[1:]):
                if point1.order_dt == point2.order_dt:
                    # logger.debug("%s %f %s ignore", point2.order_dt, point2.order_price, point2.action)
                    continue
                # logger.debug("%s %f -> %s %f %d",
                #              point1.order_dt, point1.order_price, point2.order_dt, point2.order_price, point2.action)
                # ax.plot([point1.order_dt, point2.order_dt], [point1.order_price, point2.order_price],
                #         c='r' if point2.direction != Direction.Long.value else 'g')
                if point2.direction != Direction.Long:
                    data_dict[(md_agent_key, period, symbol)]['buy_sell_point_pair'].append(
                        ([point1.order_dt, point2.order_dt], [point1.order_price, point2.order_price])
                    )
                else:
                    data_dict[(md_agent_key, period, symbol)]['sell_buy_point_pair'].append(
                        ([point1.order_dt, point2.order_dt], [point1.order_price, point2.order_price])
                    )

    # show
    show_plot_data(data_dict, title=f"[{stg_run_id}] MD and Order figure")
    return data_dict


def show_trade(stg_run_id) -> defaultdict(lambda: defaultdict(list)):
    """
    plot candle and buy and sell point
    :param stg_run_id:
    :return:
    """
    # stg_run_id=1
    stg_handler = stategy_handler_loader(stg_run_id,
                                         module_name_replacement_if_main='ibats_common.example.ma_cross_stg')
    # 加载数据库 engine
    engine_ibats = engines.engine_ibats
    # 获取全部订单
    # session = get_db_session(engine_ibats)
    with with_db_session(engine_ibats) as session:
        data_list_tot = session.query(
            TradeDetail
        ).filter(
            TradeDetail.stg_run_id == stg_run_id
        ).all()
    # 根据 md_agent 进行分组
    md_agent_key_order_detail_list_dic = defaultdict(list)
    for num, detail in enumerate(data_list_tot):
        md_agent_key = stg_handler.stg_base._td_md_agent_key_map[detail.trade_agent_key]
        md_agent_key_order_detail_list_dic[md_agent_key].append(detail)

    # 获取历史行情数据
    md_agent_key_cor_func_dic = stg_handler.get_periods_history_iterator()
    agent_count = len(md_agent_key_cor_func_dic)
    data_dict = defaultdict(lambda: defaultdict(list))
    # 根据 md_agent 对每一组行情 以及 对应的 order_detail_list 进行 plot
    # fig = plt.figure(1, figsize=(20, 4.8 * agent_count))
    for num, ((md_agent_key, period), (cor_func, meta_dic)) in enumerate(md_agent_key_cor_func_dic.items(), start=1):
        detail_list = md_agent_key_order_detail_list_dic[md_agent_key]
        df = pd.DataFrame([md_s for num, datetime_tag, md_s in cor_func])
        if df.shape[0] == 0:
            continue
        # ax = fig.add_subplot(num, 1, 1)
        # 行情
        symbol_key = meta_dic['symbol_key']
        close_key = meta_dic['close_key']
        timestamp_key = meta_dic['timestamp_key']
        for symbol, df_by_symbol in df.groupby(symbol_key):
            # df_by_symbol.set_index(timestamp_key)[close_key].plot(ax=ax, colormap='jet')
            data_dict[(md_agent_key, period, symbol)]['md'].append(df_by_symbol.set_index(timestamp_key)[close_key])
            # 开仓
            detail_list_sub = [_ for _ in detail_list
                               if _.symbol == symbol
                               and ((_.direction == Direction.Long.value and _.action == Action.Open.value)
                                    or (_.direction == Direction.Short.value and _.action != Action.Open.value)
                                    )
                               ]
            trade_date_list = [_.trade_dt for _ in detail_list_sub]
            price = [_.trade_price for _ in detail_list_sub]
            # ax.scatter(trade_date_list, price, c='r', marker='^')
            data_dict[(md_agent_key, period, symbol)]['long_open_or_short_close'].append((trade_date_list, price))
            # 关仓
            detail_list_sub = [_ for _ in detail_list
                               if (_.direction == Direction.Long.value and _.action != Action.Open.value)
                               or (_.direction == Direction.Short.value and _.action == Action.Open.value)]
            trade_date_list = [_.order_dt for _ in detail_list_sub]
            price = [_.trade_price for _ in detail_list_sub]
            # ax.scatter(trade_date_list, price, c='g', marker='v')
            data_dict[(md_agent_key, period, symbol)]['short_open_or_long_close'].append((trade_date_list, price))
            # 建立连线
            detail_list_symbol = [_ for _ in detail_list if _.symbol == symbol]
            for point1, point2 in zip(detail_list_symbol[:-1], detail_list_symbol[1:]):
                if point1.trade_dt == point2.trade_dt:
                    # logger.debug("%s %f %s ignore", point2.order_dt, point2.order_price, point2.action)
                    continue
                # logger.debug("%s %f -> %s %f %d",
                #              point1.order_dt, point1.order_price, point2.order_dt, point2.order_price, point2.action)
                # ax.plot([point1.order_dt, point2.order_dt], [point1.order_price, point2.order_price],
                #         c='r' if point2.direction != Direction.Long.value else 'g')
                if point2.direction != Direction.Long:
                    data_dict[(md_agent_key, period, symbol)]['buy_sell_point_pair'].append(
                        ([point1.trade_dt, point2.trade_dt], [point1.trade_price, point2.trade_price])
                    )
                else:
                    data_dict[(md_agent_key, period, symbol)]['sell_buy_point_pair'].append(
                        ([point1.trade_dt, point2.trade_dt], [point1.trade_price, point2.trade_price])
                    )

    # show
    show_plot_data(data_dict, title=f"[{stg_run_id}] MD and Trade figure")
    return data_dict


def show_plot_data(data_dict: dict, title=None):
    """
    将数据plot展示出来
    :param data_dict:
    :param title:
    :return:
    """
    data_len = len(data_dict)
    # fig = plt.figure(1, figsize=(20, 4.8 * data_len))
    fig, axs = plt.subplots(
        data_len, 1,
        constrained_layout=True, figsize=(20, 4.8 * data_len))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    for num, ((md_agent_key, period, symbol), plot_data_dic) in enumerate(data_dict.items()):
        # ax = fig.add_subplot(num, 1, 1)
        ax = axs[num] if data_len > 1 else axs
        ax.set_title(f"md_agent_key:{md_agent_key} - period:{period} - symbol:{symbol}")
        for md in plot_data_dic['md']:
            md.plot(ax=ax, colormap='jet')
        for x, y in plot_data_dic['long_open_or_short_close']:
            ax.scatter(x, y, c='r', marker='^')
        for x, y in plot_data_dic['short_open_or_long_close']:
            ax.scatter(x, y, c='g', marker='v')
        for x, y in plot_data_dic['buy_sell_point_pair']:
            ax.plot(x, y, c='r')
        for x, y in plot_data_dic['sell_buy_point_pair']:
            ax.plot(x, y, c='g')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    stg_run_id = None
    show_cash_and_margin(stg_run_id)
    # data_dict = show_order(stg_run_id, module_name_replacement_if_main='ibats_common.example.ai_stg')
