#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/10/6 10:52
@File    : xy.py
@contact : mmmaaaggg@163.com
@desc    : 生成 X Y 数据并进行筛选，对齐等
"""
import logging
import pandas as pd
import numpy as np
from ibats_common.backend.factor import get_factor
import ibats_common.config  # NOQA

logger = logging.getLogger()


def get_y(df: pd.DataFrame, target_n_bars, rolling_label='close_price', calc_func=lambda x: x.calc_total_return()):
    """
    计算Y标签数值
    :param df:
    :param target_n_bars:
    :param rolling_label:
    :param calc_func:
    :return:
    """
    y_s = df[rolling_label].rolling(
        window=target_n_bars
    ).apply(
        calc_func
    ).iloc[target_n_bars:]  # calc_total_return calc_calmar_ratio
    y_s.name = 'Y'
    return y_s


def get_xy(df: pd.DataFrame, target_n_bars=5, get_factor_kwargs=None, get_y_kwargs=None):
    """
    生成 X Y 数据并进行筛选，对齐等
    :param df:
    :param target_n_bars:
    :param get_factor_kwargs:
    :param get_y_kwargs:
    :return:
    """
    get_factor_kwargs = {} if get_factor_kwargs is None else get_factor_kwargs
    factor_df = get_factor(df, **get_factor_kwargs)
    get_y_kwargs = {} if get_y_kwargs is None else get_y_kwargs
    get_y_kwargs.update({"target_n_bars": target_n_bars})
    y_s = get_y(df, **get_factor_kwargs)
    # 数据切片
    hist_bar_df, factor_df = df.iloc[:-target_n_bars], factor_df.iloc[:-target_n_bars]
    logger.info("hist_bar_df.shape=%s, factor_df.shape=%s", hist_bar_df.shape, factor_df.shape)
    logger.info("y_s.shape=%s", y_s.shape)
    assert factor_df.shape[0] == y_s.shape[0], \
        f"因子数据 x 长度 {factor_df.shape[0]} 要与训练目标 y 数据长度 {y_s.shape[0]} 一致"
    # 剔除无效数据，并根据 target_n_bars 进行数据切片
    is_available = ~(np.isinf(y_s.to_numpy())
                     | np.isnan(y_s.to_numpy())
                     | np.any(np.isnan(factor_df.to_numpy()), axis=1)
                     | np.any(np.isinf(factor_df.to_numpy()), axis=1))
    available_df = hist_bar_df[is_available]
    available_factor_df = factor_df[is_available]
    x_arr = available_factor_df.to_numpy()
    y_arr = y_s[is_available]
    assert x_arr.shape[0] == y_arr.shape[0], \
        f"因子数据 x 长度 {x_arr.shape[0]} 要与训练目标 y 数据长度 {y_arr.shape[0]} 一致"
    logger.info("x_arr.shape=%s, y_arr.shape=%s", x_arr.shape, y_arr.shape)
    return available_df, available_factor_df, x_arr, y_arr


if __name__ == "__main__":
    pass
