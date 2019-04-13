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
from ibats_utils.db import with_db_session, get_db_session
from ibats_common.backend import engines
import pandas as pd
from ibats_common.backend.orm import StgRunStatusDetail, OrderDetail
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
    # session = get_db_session(engine_ibats)
    with with_db_session(engine_ibats) as session:
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


def show_order(stg_run_id):
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
    # 根据 md_agent 对每一组行情 以及 对应的 order_detail_list 进行 plot
    fig = plt.figure(1, figsize=(20, 4.8 * agent_count))
    for num, ((md_agent_key, period), cor_func) in enumerate(md_agent_key_cor_func_dic.items(), start=1):
        order_detail_list = md_agent_key_order_detail_list_dic[md_agent_key]
        df = pd.DataFrame([md_s for num, datetime_tag, md_s in cor_func])
        if df.shape[0] == 0:
            continue
        ax = fig.add_subplot(num, 1, 1)
        # 行情
        df.set_index('trade_date')['close'].plot(ax=ax, colormap='jet')
        # 开仓
        order_detail_list_sub = [_ for _ in order_detail_list
                                 if (_.direction == Direction.Long.value and _.action == Action.Open.value)
                                 or (_.direction == Direction.Short.value and _.action != Action.Open.value)]
        trade_date_list = [_.order_dt for _ in order_detail_list_sub]
        price = [_.order_price for _ in order_detail_list_sub]
        ax.scatter(trade_date_list, price, c='r', marker='^')
        # 关仓
        order_detail_list_sub = [_ for _ in order_detail_list
                                 if (_.direction == Direction.Long.value and _.action != Action.Open.value)
                                 or (_.direction == Direction.Short.value and _.action == Action.Open.value)]
        trade_date_list = [_.order_dt for _ in order_detail_list_sub]
        price = [_.order_price for _ in order_detail_list_sub]
        ax.scatter(trade_date_list, price, c='g', marker='v')
        # 建立连线
        for point1, point2 in zip(order_detail_list[:-1], order_detail_list[1:]):
            if point1.order_dt == point2.order_dt:
                # logger.debug("%s %f %s ignore", point2.order_dt, point2.order_price, point2.action)
                continue
            ax.plot([point1.order_dt, point2.order_dt], [point1.order_price, point2.order_price],
                    c='r' if point2.direction != Direction.Long.value else 'g')
            # logger.debug("%s %f -> %s %f %d",
            #              point1.order_dt, point1.order_price, point2.order_dt, point2.order_price, point2.action)

    # show
    fig.show()
    plt.close(fig)
