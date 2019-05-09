#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from scipy import stats
import numpy as np
import ffn


logger = logging.getLogger(__name__)


def drawdown_plot(df: pd.DataFrame, drawdown_col_name_list):
    """
    回撤曲线
    :param df:
    :param drawdown_col_name_list:
    :return:
    """
    if drawdown_col_name_list is None:
        data_df = df.to_drawdown_series()
    else:
        data_df = df[drawdown_col_name_list].to_drawdown_series()

    data_df.plot()
    plt.show()


def hist_norm(data, bins=10, title=None):
    """hist 分布图及正太分布曲线"""
    # if title is None:
    #     title = r'hist bar'
    # ax = pct_change_s.hist(bins=50, density=1)
    fig, ax = plt.subplots()
    # the histogram of the data
    _, _, patches = ax.hist(data, bins, density=True)
    n, bins_v = plot_norm(data, bins=bins, ax=ax)
    # ax.set_xlabel('pct change')
    # ax.set_ylabel('change rate')
    if title is not None:
        ax.set_title(title)
    plt.show()
    return n, bins_v


def plot_norm(data: pd.Series, bins=10, ax=None, is_show_plot=None):
    """
    显示当前数据的正太分布曲线
    :param data:
    :param bins: bar 数量
    :param ax: 如果为None，则新建一个画布
    :param is_show_plot: 是否展示
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()
        if is_show_plot is None:
            is_show_plot = True

    if is_show_plot is None:
        is_show_plot = False

    n, bins_v = np.histogram(data, bins=bins)

    mu = data.mean()  # mean of distribution
    sigma = data.std()  # standard deviation of distribution
    # def norm_func(x, mu, sigma):
    #     pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    #     return pdf
    # y = norm_func(bins, mu, sigma)  # 与 mlab.normpdf(bins, mu, sigma) 相同
    # y = mlab.normpdf(bins, mu, sigma)
    y = stats.norm.pdf(bins, loc=mu, scale=sigma)
    ax.plot(bins, y, '--')
    if is_show_plot:
        plt.show()

    return n, bins_v


def hist_norm_sns(data, bins=10):
    """sns 方式 hist 分布图及正太分布曲线"""
    sns.set_palette("hls")
    sns.distplot(
        data, bins=bins,
        fit=stats.norm,
        kde_kws={"color": "darkorange", "lw": 1, "label": "KDE", "linestyle": "--"},
        hist_kws={"color": "darkblue"})
    plt.show()


