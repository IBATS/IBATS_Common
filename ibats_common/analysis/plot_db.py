#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-9 下午4:34
@File    : plot_db.py
@contact : mmmaaaggg@163.com
@desc    : 需要依赖于数据库进行输出展示的函数
"""
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from ibats_utils.db import with_db_session
from ibats_utils.mess import date_2_str
from sqlalchemy.sql import func

from ibats_common.backend.mess import get_cache_folder_path
from ibats_common.analysis.plot import get_file_name, plot_or_show
from ibats_common.backend import engines
from ibats_common.backend.mess import get_stg_run_id_latest
from ibats_common.backend.orm import StgRunStatusDetail, OrderDetail, TradeDetail, StgRunInfo
from ibats_common.common import Action, Direction, RunMode
from ibats_common.strategy_handler import strategy_handler_loader

logger = logging.getLogger(__name__)


def show_cash_and_margin(stg_run_id, enable_show_plot=True, enable_save_plot=False, run_mode=RunMode.Backtest, **kwargs):
    """
    plot cash_and_margin
    :param stg_run_id:
    :param enable_show_plot:
    :param enable_save_plot:
    :param run_mode:
    :param kwargs:
    :return:
    """
    # stg_run_id=154
    engine_ibats = engines.engine_ibats
    with with_db_session(engine_ibats) as session:
        if stg_run_id is None:
            stg_run_id = session.query(func.max(StgRunInfo.stg_run_id)).scalar()
            logger.warning('没有设置 stg_run_id 参数，将输出最新的 stg_run_id=%d 对应记录', stg_run_id)

        sql_str = str(
            session.query(
                StgRunStatusDetail.trade_dt.label('trade_dt'),
                StgRunStatusDetail.cash_available.label('cash'),
                (StgRunStatusDetail.cash_available.label('cash') + StgRunStatusDetail.commission_tot
                 ).label('cash + commission'),
                StgRunStatusDetail.curr_margin.label('margin'),
                StgRunStatusDetail.cash_and_margin.label('cash_and_margin'),
                (StgRunStatusDetail.cash_and_margin.label('cash_and_margin') + StgRunStatusDetail.commission_tot
                 ).label('no commission'),
            ).filter(
                StgRunStatusDetail.stg_run_id == stg_run_id
            )
        )

    df = pd.read_sql(sql_str, engine_ibats, params=[stg_run_id], index_col=['trade_dt'])
    if df.shape[0] == 0:
        file_path = None
        return df, file_path

    ax = df[['cash', 'margin']].plot.area()
    if run_mode != RunMode.Backtest_FixPercent:
        df[['cash_and_margin', 'no commission', 'cash + commission']].plot(ax=ax)

    ax.set_title(
        f"Cash + Margin [{stg_run_id}] "
        f"{date_2_str(min(df.index))} - {date_2_str(max(df.index))} ({df.shape[0]} days)")

    if enable_save_plot:
        file_name = get_file_name(f'cash_and_margin', name=stg_run_id)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()

    return df, file_path


def get_md(stg_run_id, return_td_md_agent_key_map=False):
    # 获取行情数据
    stg_handler = strategy_handler_loader(stg_run_id, is_4_shown=True)

    td_md_agent_key_map = stg_handler.stg_base._td_md_agent_key_map

    # 获取历史行情数据
    md_agent_key_cor_func_dic = stg_handler.get_periods_history_iterator()
    # 根据 md_agent 对每一组行情 以及 对应的 order_detail_list 进行 plot
    # fig = plt.figure(1, figsize=(20, 4.8 * agent_count))
    symbol_rr_dic = {}
    sum_rr_df, sum_col_name_list = None, []
    for num, ((md_agent_key, period), (cor_func, meta_dic)) in enumerate(md_agent_key_cor_func_dic.items(), start=1):
        df = pd.DataFrame([md_s for num, datetime_tag, md_s in cor_func])
        if df.shape[0] == 0:
            continue
        # ax = fig.add_subplot(num, 1, 1)
        # 行情
        symbol_key = meta_dic['symbol_key']
        close_key = meta_dic['close_key']
        timestamp_key = meta_dic['timestamp_key']
        df.set_index(timestamp_key, inplace=True)
        for symbol, df_by_symbol in df.groupby(symbol_key):
            symbol_rr_dic[(md_agent_key, period, symbol)] = (df_by_symbol, close_key)

            # 汇总输出
            col_name = f"{symbol}_rr"
            sum_col_name_list.append(col_name)
            md_df = df_by_symbol[[close_key]].copy()
            md_df['md_rr'] = md_df[close_key] / md_df[close_key].iloc[0]
            if sum_rr_df is None:
                sum_rr_df = md_df.rename(columns={'md_rr': col_name})[sum_col_name_list]
            else:
                sum_rr_df = sum_rr_df.join(md_df.rename(columns={'md_rr': col_name}))[sum_col_name_list]

    if return_td_md_agent_key_map:
        return sum_rr_df, symbol_rr_dic, td_md_agent_key_map
    else:
        return sum_rr_df, symbol_rr_dic


def get_rr_with_md(stg_run_id, compound_rr=True):
    """
    获取策略收益率数据
    :param stg_run_id:
    :param compound_rr:复合收益率
    :return:
    """
    engine_ibats = engines.engine_ibats
    # 获取 收益曲线
    with with_db_session(engine_ibats) as session:
        if compound_rr:
            sql_str = str(
                session.query(
                    StgRunStatusDetail.trade_dt.label('trade_dt'),
                    StgRunStatusDetail.cash_and_margin.label('cash and margin'),
                    (StgRunStatusDetail.cash_and_margin.label('cash_and_margin') +
                     StgRunStatusDetail.commission_tot.label('commission_tot')).label('no_commission'),
                    StgRunStatusDetail.rr_compound.label('rr'),
                    StgRunStatusDetail.rr_compound_nc.label('rr no commission'),
                ).filter(
                    StgRunStatusDetail.stg_run_id == stg_run_id
                )
            )
        else:
            sql_str = str(
                session.query(
                    StgRunStatusDetail.trade_dt.label('trade_dt'),
                    StgRunStatusDetail.cash_and_margin.label('cash and margin'),
                    (StgRunStatusDetail.cash_and_margin.label('cash_and_margin') +
                     StgRunStatusDetail.commission_tot.label('commission_tot')).label('no_commission'),
                    StgRunStatusDetail.rr.label('rr'),
                    StgRunStatusDetail.rr_nc.label('rr no commission'),
                ).filter(
                    StgRunStatusDetail.stg_run_id == stg_run_id
                )
            )

    rr_df = pd.read_sql(sql_str, engine_ibats, params=[stg_run_id]).set_index('trade_dt')
    if rr_df is None or rr_df.shape[0] == 0:
        return None, None

    # rr_df['rr'] = rr_df['cash and margin'] / rr_df['cash and margin'].iloc[0]
    # rr_df['rr without commission'] = rr_df['without commission'] / rr_df['without commission'].iloc[0]
    col_list_rr = ['rr', 'rr no commission']
    rr_df[col_list_rr] += 1

    # 获取行情数据
    sum_df, symbol_rr_dic = get_md(stg_run_id)
    sum_df = sum_df.join(rr_df[col_list_rr])

    col_list = ['md_rr']
    col_list.extend(col_list)
    for num, (key, (df, close_key)) in enumerate(symbol_rr_dic.items()):
        md_df = df[[close_key]].copy()
        md_df['md_rr'] = md_df[close_key] / md_df[close_key].iloc[0]
        md_df = md_df.join(rr_df)[col_list]
        symbol_rr_dic[key] = (md_df, close_key)

    return sum_df, symbol_rr_dic


def show_rr_with_md(stg_run_id, show_sum_plot=True, show_each_md_plot=False, enable_save_plot=False):
    if stg_run_id is None:
        stg_run_id = get_stg_run_id_latest()

    sum_df, symbol_rr_dic = get_rr_with_md(stg_run_id)
    save_file_path_dic = {}

    if show_each_md_plot:
        for num, (symbol, (df, close_key)) in enumerate(symbol_rr_dic.items(), start=1):
            ax = df.plot()
            ax.set_title(
                f"Return Rate {symbol} [{stg_run_id}] "
                f"{date_2_str(min(df.index))} - {date_2_str(max(df.index))} ({df.shape[0]} days)")
            plt.show()

    if show_sum_plot:
        ax = sum_df.plot()
        ax.set_title(
            f"Return Rate [{stg_run_id}] "
            f"{date_2_str(min(sum_df.index))} - {date_2_str(max(sum_df.index))} ({sum_df.shape[0]} days)")
        plt.show()

    if enable_save_plot:
        ax = sum_df.plot()
        ax.set_title(
            f"Return Rate [{stg_run_id}] "
            f"{date_2_str(min(sum_df.index))} - {date_2_str(max(sum_df.index))} ({sum_df.shape[0]} days)")

        rr_plot_file_path = os.path.join(get_cache_folder_path(), f'rr_plot {stg_run_id}.png')
        plt.savefig(rr_plot_file_path, dpi=75)
        save_file_path_dic['rr'] = rr_plot_file_path

    return sum_df, symbol_rr_dic, save_file_path_dic


def show_order(stg_run_id, **kwargs) -> (defaultdict(lambda: defaultdict(list)), str):
    """
    plot candle and buy and sell point
    :param stg_run_id:
    :param kwargs:
    :return:
    """
    # 加载数据库 engine
    engine_ibats = engines.engine_ibats
    # stg_run_id = 1
    if stg_run_id is None:
        stg_run_id = get_stg_run_id_latest()

    # 获取行情数据
    sum_rr_df, symbol_rr_dic, td_md_agent_key_map = get_md(stg_run_id, return_td_md_agent_key_map=True)

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
    for num, detail in enumerate(order_detail_list_tot):
        md_agent_key = td_md_agent_key_map[detail.trade_agent_key]
        md_agent_key_order_detail_list_dic[md_agent_key].append(detail)

    # 获取历史行情数据
    data_dict = defaultdict(lambda: defaultdict(list))
    for num, ((md_agent_key, period, symbol), (df_by_symbol, close_key)) in enumerate(symbol_rr_dic.items(), start=1):
        detail_list = md_agent_key_order_detail_list_dic[md_agent_key]
        data_dict[(md_agent_key, period, symbol)]['md'].append(df_by_symbol[close_key])
        # 开仓
        order_detail_list_sub = [_ for _ in detail_list
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
        order_detail_list_sub = [_ for _ in detail_list
                                 if (_.direction == Direction.Long.value and _.action != Action.Open.value)
                                 or (_.direction == Direction.Short.value and _.action == Action.Open.value)]
        trade_date_list = [_.order_dt for _ in order_detail_list_sub]
        price = [_.order_price for _ in order_detail_list_sub]
        # ax.scatter(trade_date_list, price, c='g', marker='v')
        data_dict[(md_agent_key, period, symbol)]['short_open_or_long_close'].append((trade_date_list, price))
        # 建立连线
        order_detail_list_symbol = [_ for _ in detail_list if _.symbol == symbol]
        for point1, point2 in zip(order_detail_list_symbol[:-1], order_detail_list_symbol[1:]):
            if point1.order_dt == point2.order_dt:
                # self.logger.debug("%s %f %s ignore", point2.order_dt, point2.order_price, point2.action)
                continue
            # self.logger.debug("%s %f -> %s %f %d",
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
    file_path = show_plot_data_dic(data_dict, title=f"MD and Order figure [{stg_run_id}]",
                                   stg_run_id=stg_run_id, **kwargs)
    return data_dict, file_path


def show_trade(stg_run_id, **kwargs) -> (defaultdict(lambda: defaultdict(list)), str):
    """
    plot candle and buy and sell point
    :param stg_run_id:
    :param kwargs:
    :return:
    """
    # stg_run_id = 1
    if stg_run_id is None:
        stg_run_id = get_stg_run_id_latest()

    # 获取行情数据
    sum_rr_df, symbol_rr_dic, td_md_agent_key_map = get_md(stg_run_id, return_td_md_agent_key_map=True)

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
        md_agent_key = td_md_agent_key_map[detail.trade_agent_key]
        md_agent_key_order_detail_list_dic[md_agent_key].append(detail)

    # 获取历史行情数据
    data_dict = defaultdict(lambda: defaultdict(list))
    for num, ((md_agent_key, period, symbol), (df_by_symbol, close_key)) in enumerate(symbol_rr_dic.items(), start=1):
        detail_list = md_agent_key_order_detail_list_dic[md_agent_key]
        data_dict[(md_agent_key, period, symbol)]['md'].append(df_by_symbol[close_key])
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
        trade_date_list = [_.trade_dt for _ in detail_list_sub]
        price = [_.trade_price for _ in detail_list_sub]
        # ax.scatter(trade_date_list, price, c='g', marker='v')
        data_dict[(md_agent_key, period, symbol)]['short_open_or_long_close'].append((trade_date_list, price))
        # 建立连线
        detail_list_symbol = [_ for _ in detail_list if _.symbol == symbol]
        for point1, point2 in zip(detail_list_symbol[:-1], detail_list_symbol[1:]):
            if point1.trade_dt == point2.trade_dt:
                # self.logger.debug("%s %f %s ignore", point2.order_dt, point2.order_price, point2.action)
                continue
            # self.logger.debug("%s %f -> %s %f %d",
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
    file_path = show_plot_data_dic(data_dict, title=f"MD and Trade figure [{stg_run_id}]",
                                   stg_run_id=stg_run_id, **kwargs)
    return data_dict, file_path


def show_plot_data_dic(data_dict: dict, title=None, enable_show_plot=True, enable_save_plot=False, stg_run_id=None, **kwargs):
    """
    将数据plot展示出来
    :param data_dict:
    :param title:
    :param enable_show_plot:
    :param enable_save_plot:
    :param stg_run_id:
    :param kwargs:
    :return:
    """
    data_len = len(data_dict)

    fig, axs = plt.subplots(
        data_len, 1,
        # constrained_layout=True,
        figsize=(20, 4.8 * data_len))
    name = '' if title is None else title

    # 容易与 ax title 重叠
    # if title is not None:
    #     fig.suptitle(title, fontsize=16)
    for num, ((md_agent_key, period, symbol), plot_data_dic) in enumerate(data_dict.items()):
        # ax = fig.add_subplot(num, 1, 1)
        ax = axs[num] if data_len > 1 else axs
        ax.set_title(f"{name} - md_agent_key={md_agent_key} - period={period} - symbol={symbol}")
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

    file_name = get_file_name(f'title', name=title)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                             file_name=file_name, stg_run_id=stg_run_id)

    return file_path


def _test_show_rr_with_md():
    stg_run_id = None
    # data_dict = show_order(
    #     stg_run_id,
    #     # module_name_replacement_if_main='ibats_common.example.tf_stg.ai_stg',
    #     module_name_replacement_if_main='ibats_common.example.ma_cross_stg',
    # )
    # show_cash_and_margin(stg_run_id)
    show_rr_with_md(stg_run_id, show_each_md_plot=True, show_sum_plot=True, enable_save_plot=True)


def _test_show_cash_and_margin():
    stg_run_id = None
    show_cash_and_margin(stg_run_id, enable_show_plot=True, enable_save_plot=True)


if __name__ == '__main__':
    # _test_show_rr_with_md()
    _test_show_cash_and_margin()
