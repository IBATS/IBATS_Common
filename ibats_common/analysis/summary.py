#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-9 上午9:53
@File    : summary.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from scipy.stats import anderson, normaltest
from collections import defaultdict
from ibats_common.analysis.corr import corr
from ibats_common.analysis.plot import drawdown_plot
from ibats_common.analysis.wave import wave_hist
import pandas as pd
import numpy as np
import ffn
import logging

logger = logging.getLogger(__name__)


def summary(df: pd.DataFrame, percentiles=[0.2, 1 / 3, 0.5, 2 / 3, 0.8], risk_free=0.03,
            figure_4_each_col=False,
            col_transfer_dic: (dict, None) = None,
            stat_col_name_list=None,
            drawdown_col_name_list=None,
            ):
    """
    汇总展示数据分析结果，同时以 dict 形式返回各项指标分析结果
    第一个返回值，df的各项分析结果
    第二个返回值，各个列的各项分析结果
    :param df:
    :param percentiles:分为数信息
    :param risk_free:无风险收益率
    :param figure_4_each_col:hist图使用，每一列显示单独一张图片
    :param col_transfer_dic:列转换方法
    :param stat_col_name_list:对哪些列的数据执行统计
    :return:
    """
    columns = list(df.columns)
    logger.info('data columns: %s', columns)
    ret_dic = {}
    each_col_dic = defaultdict(dict)

    logger.info('Description:')
    df.describe(percentiles=percentiles)
    corr_df = corr(df)
    ret_dic['corr'] = corr_df
    logger.info('Correlation Coefficient:\n%s', corr_df)
    # ffn_stats = ffn.calc_stats(df)
    # histgram 分布图
    n_bins_dic = wave_hist(df, figure_4_each_col=figure_4_each_col, col_transfer_dic=col_transfer_dic)
    ret_dic['wave_hist'] = n_bins_dic
    # 回撤曲线
    drawdown_df = drawdown_plot(df, drawdown_col_name_list)
    ret_dic['drawdown'] = drawdown_df
    # 单列分析
    stat_df = (df if stat_col_name_list is None else df[stat_col_name_list])
    for col_name, data in stat_df.items():
        data = data.dropna()
        data = data[~np.isinf(data)]
        logger.info("=" * 50 + ' %s ' + "=" * 50, col_name)
        result = normaltest(data)
        each_col_dic[col_name]['normaltest'] = result
        logger.info('%s %s', col_name, result)
        result = anderson(data)
        logger.info('%s %s', col_name, result)
        each_col_dic[col_name]['anderson'] = result
        stats = data.calc_perf_stats()
        # stats = ffn_stats[col_name]
        # 设置无风险收益率
        stats.set_riskfree_rate(risk_free)
        stats.display()

    return ret_dic, each_col_dic


if __name__ == "__main__":
    from ibats_common.example.data import load_data

    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    summary(df, drawdown_col_name_list=['close'], figure_4_each_col=False, stat_col_name_list=['close'], col_transfer_dic=col_transfer_dic)
