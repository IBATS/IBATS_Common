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

logger = logging.getLogger(__name__)


def get_factor(df: pd.DataFrame, dropna=True) -> pd.DataFrame:
    ret_df = df.copy()

    if dropna:
        ret_df.dropna(inplace=True)
    return ret_df


def _test_get_factor():
    from ibats_common.example.data import load_data

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    factor = get_factor(df)
    logger.info("\n%s\t%s \n%s\t%s", df.shape, list(df.columns), factor.shape, list(factor.columns))
    print("\n%s\t%s \n%s\t%s" % (df.shape, list(df.columns), factor.shape, list(factor.columns)))


if __name__ == '__main__':
    import ibats_common.config
    _test_get_factor()
