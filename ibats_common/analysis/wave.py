#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-8 下午2:53
@File    : wave.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
import numpy as np
from ibats_common.analysis.plot import hist_norm, plot_norm
from ibats_common.example.data import load_data
import logging
import itertools
import matplotlib.pyplot as plt
# import scipy.stats as stats
# import ffn 不可删除 DataFrame.to_returns 依赖于此导入
import ffn

logger = logging.getLogger(__name__)


def wave_hist(df: pd.DataFrame, perf_stats=None, columns=None, bins=50, figure_4_each_col=True,
              col_transfer_dic: (dict, None) = None):
    """
    波动率分布图
    :param df:
    :param perf_stats: df.calc_stats()
    :param columns: 显示哪些列
    :param bins: bar 数量
    :param figure_4_each_col: 每一个col单独一张图
    :param col_transfer_dic: 列值转换
    :return:
    """

    if columns is not None:
        data_df = df[columns].copy()
    else:
        data_df = df.copy()

    data_df.dropna(inplace=True)
    if col_transfer_dic is not None:
        for func, col_name_list in col_transfer_dic.items():
            if func == 'return':
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    rename_dic[col_name] = f"{col_name} {func}"
                    data_df.loc[:, col_name] = data_df[col_name].to_returns()
                else:
                    data_df.rename(columns=rename_dic, inplace=True)

            elif func == 'pct_change':
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    rename_dic[col_name] = f"{col_name} {func}"
                    data_df.loc[:, col_name] = data_df[col_name].pct_change()
                else:
                    data_df.rename(columns=rename_dic, inplace=True)
            else:
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    rename_dic[col_name] = f"{col_name} {str(func)}"
                    data_df.loc[:, col_name] = func(data_df[col_name])
                else:
                    data_df.rename(columns=rename_dic, inplace=True)

        data_df.dropna(inplace=True)

    columns = list(data_df.columns)
    n_bins_dic = {}
    if figure_4_each_col:
        for col_name in columns:
            data = data_df[col_name]
            try:
                data = data[~np.isinf(data)].dropna()
            except TypeError as exp:
                logger.exception('column %s 数据类型无效，无法进行 np.isinf 计算', col_name)
                raise exp from exp

            n, bins_v = hist_norm(data, bins=bins, name=col_name)
            n_bins_dic[col_name] = (n, bins_v)

    else:
        ax = data_df.hist(bins=bins)
        if isinstance(ax, np.ndarray):
            # ax 总是一个 偶数 出现，因此，需要进行一次长度对齐
            ax_list = list(itertools.chain(*ax))[:len(columns)]
        else:
            ax_list = [ax]

        for col_name, ax_sub in zip(columns, ax_list):
            # pct_change_s = df['close'].pct_change().dropna()
            # logger.info('pct_change description:\n%s', pct_change_s.describe())
            # logger.info('pct_change quantile:\n%s', pct_change_s.quantile([_ / 20 for _ in range(20)]))
            data = data_df[col_name]
            try:
                data = data[~np.isinf(data)].dropna()
            except TypeError as exp:
                logger.exception('column %s 数据类型无效，无法进行 np.isinf 计算', col_name)
                raise exp from exp

            n, bins_v, mean, std = plot_norm(data, bins=bins, ax=ax_sub)
            # 字太多无法显示
            # ax_sub.set_title(f"{col_name}\n(mean={mean:.4f} std={std:.4f})")
            n_bins_dic[col_name] = (n, bins_v)

        plt.show()

    return n_bins_dic


if __name__ == "__main__":
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    n_bins_dic = wave_hist(df, figure_4_each_col=False, col_transfer_dic=col_transfer_dic)
