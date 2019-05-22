#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/5/20 21:09
@File    : factor.py
@contact : mmmaaaggg@163.com
@desc    : 用来对时间序列数据建立因子
"""
import pandas as pd
import logging
import datetime
import numpy as np


logger = logging.getLogger(__name__)


def add_factor_of_trade_date(df: pd.DataFrame, trade_date_series):
    index_s = pd.Series(df.index)

    # 本月第几个交易日, 本月还剩几个交易日
    groups = trade_date_series.groupby(trade_date_series.apply(lambda x: datetime.datetime(x.year, x.month, 1)))

    def get_td_of_month(dt):
        first_day_of_month = datetime.datetime(dt.year, dt.month, 1)
        month_s = groups.get_group(first_day_of_month)
        td_of_month = (month_s <= dt).sum()
        td_left_of_month = (month_s > dt).sum()
        return month_s.shape[0], td_of_month, td_left_of_month

    result_arr = index_s.apply(get_td_of_month).to_numpy()
    df['td_of_month'] = [_[0] for _ in result_arr]
    df['td_pass_of_month'] = [_[1] for _ in result_arr]
    df['td_left_of_month'] = [_[2] for _ in result_arr]

    # 本周第几个交易日, 本周还剩几个交易日
    groups = trade_date_series.groupby(trade_date_series.apply(lambda x: x.year * 100 + x.weekofyear))

    def get_td_of_week(dt):
        name = dt.year * 100 + dt.weekofyear
        week_s = groups.get_group(name)
        td_pass_of_week = (week_s <= dt).sum()
        td_left_of_week = (week_s > dt).sum()
        return week_s.shape[0], td_pass_of_week, td_left_of_week

    result_arr = index_s.apply(get_td_of_week).to_numpy()
    df['td_of_week'] = [_[0] for _ in result_arr]
    df['td_pass_of_week'] = [_[1] for _ in result_arr]
    df['td_left_of_week'] = [_[2] for _ in result_arr]

    # 距离下一次放假交易日数（超过2天以上的休息日）
    # 计算距离下一个交易日的日期
    days_2_next_trade_date_s = (index_s.shift(-1) - index_s).fillna(pd.Timedelta(days=0))
    # 倒序循环，计算距离下一次放假的日期
    days_count, trade_date_count, result_arr = np.nan, np.nan, []
    for _, delta in days_2_next_trade_date_s.sort_index(ascending=False).items():
        days = delta.days
        if days > 3:
            trade_date_count, days_count = 0, 0
        elif pd.isna(days_count) >= 0 and days >= 1:
            trade_date_count += 1
            days_count += days
        else:
            trade_date_count, days_count = np.nan, np.nan

        result_arr.append([days_count, trade_date_count])

    result_arr.reverse()
    df['days_2_vacation'] = [_[0] for _ in result_arr]
    df['td_2_vacation'] = [_[1] for _ in result_arr]

    return df


def add_factor_of_delivery_date(df, delivery_date_series):
    return df


def get_factor(df: pd.DataFrame, trade_date_series=None, delivery_date_series=None, dropna=True) -> pd.DataFrame:
    ret_df = df.copy()
    # 自然日相关因子
    index_s = pd.Series(ret_df.index)
    ret_df['dayofweek'] = index_s.apply(lambda x: x.dayofweek).to_numpy()
    ret_df['day'] = index_s.apply(lambda x: x.day).to_numpy()
    ret_df['month'] = index_s.apply(lambda x: x.month).to_numpy()
    ret_df['daysleftofmonth'] = index_s.apply(lambda x: x.days_in_month - x.day).to_numpy()

    # 交易日相关因子
    if trade_date_series is not None:
        ret_df = add_factor_of_trade_date(ret_df, trade_date_series)

    # 交割日相关因子
    if delivery_date_series is not None:
        ret_df = add_factor_of_delivery_date(ret_df, delivery_date_series)
        # 当期合约距离交割日天数

    if dropna:
        ret_df.dropna(inplace=True)
    return ret_df


def _test_get_factor():
    from ibats_common.example.data import load_data, get_trade_date_series

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    factor = get_factor(df, trade_date_series=get_trade_date_series())
    logger.info("\n%s\t%s \n%s\t%s", df.shape, list(df.columns), factor.shape, list(factor.columns))
    print("\n%s\t%s \n%s\t%s" % (df.shape, list(df.columns), factor.shape, list(factor.columns)))


if __name__ == '__main__':
    import ibats_common.config
    _test_get_factor()
