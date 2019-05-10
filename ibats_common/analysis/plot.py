#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from scipy import stats
import numpy as np
import ffn
from ibats_utils.mess import date_2_str
from ibats_common.analysis import get_cache_folder_path

logger = logging.getLogger(__name__)


def drawdown_plot(df: pd.DataFrame, perf_stats=None, col_name_list=None, enable_show_plot=True, enable_save_plot=False, name=None):
    """
    回撤曲线
    :param df:
    :param perf_stats:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :return:
    """
    if col_name_list is None:
        data_df = df.to_drawdown_series()
    else:
        data_df = df[col_name_list].to_drawdown_series()

    if perf_stats is None:
        perf_stats = df.calc_stats()

    col_mdd_len_dic = {col_name: s.drawdown_details.Length.max() for col_name, s in perf_stats.items()}
    col_mdd_dic = {col_name: mdd for col_name, mdd in data_df.min().items()}
    data_df.rename(
        columns={col_name: f"{col_name}[{mdd*100:.2f}% {col_mdd_len_dic[col_name]}]"
                 for col_name, mdd in col_mdd_dic.items()},
        inplace=True)

    if enable_show_plot:
        ax = data_df.plot()
        ax.set_title(f"Drawdown {['{:.2f}%'.format(col_mdd_dic[_] * 100) for _ in df.columns]}")
        plt.show()

    if enable_save_plot:
        ax = data_df.plot()
        ax.set_title(f"Drawdown {['{:.2f}%'.format(col_mdd_dic[_] * 100) for _ in df.columns]}")
        file_name = f'drawdown {np.random.randint(10000)}.png' if name is None else f'drawdown {name}.png'
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    return data_df, file_path


def plot_rr_df(df: pd.DataFrame, col_name_list=None, enable_show_plot=True, enable_save_plot=False, name=None):
    """
    Return Rate
    :param df:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :return:
    """
    if col_name_list is not None:
        data_df = df[col_name_list]
    else:
        data_df = df

    if enable_show_plot:
        ax = data_df.plot()
        ax.set_title(
            f"Return Rate " if name is None else f"Return Rate [{name}] "  
            f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")
        plt.show()

    if enable_save_plot:
        ax = data_df.plot()
        ax.set_title(
            f"Return Rate " if name is None else f"Return Rate [{name}] "  
            f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

        file_name = f'rr_plot {np.random.randint(10000)}.png' if name is None else f'rr_plot {name}.png'
        rr_plot_file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(rr_plot_file_path, dpi=75)
    else:
        rr_plot_file_path = None

    return rr_plot_file_path


def hist_norm(data, bins=10, name=None):
    """hist 分布图及正太分布曲线"""
    # if title is None:
    #     title = r'hist bar'
    # ax = pct_change_s.hist(bins=50, density=1)
    fig, ax = plt.subplots()
    # the histogram of the data
    _, _, patches = ax.hist(data, bins, density=True)
    n, bins_v, mean, std = plot_norm(data, bins=bins, ax=ax)
    # ax.set_xlabel('pct change')
    # ax.set_ylabel('change rate')
    ax.set_title(f"{'Data' if name is None else name} Histogram (mean={mean:.4f} std={std:.4f})")
    plt.show()
    return n, bins_v


def plot_norm(data: pd.Series, bins=10, ax=None, is_show_plot=None):
    """
    显示当前数据的正太分布曲线
    :param data:
    :param bins: bar 数量
    :param ax: 如果为None，则新建一个画布
    :param is_show_plot: 是否展示
    :return: n, bins_v, mean, std
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
    y = stats.norm.pdf(bins_v, loc=mu, scale=sigma)
    ax.plot(bins_v, y, '--')
    if is_show_plot:
        plt.show()

    return n, bins_v, mu, sigma


def hist_norm_sns(data, bins=10):
    """sns 方式 hist 分布图及正太分布曲线"""
    sns.set_palette("hls")
    sns.distplot(
        data, bins=bins,
        fit=stats.norm,
        kde_kws={"color": "darkorange", "lw": 1, "label": "KDE", "linestyle": "--"},
        hist_kws={"color": "darkblue"})
    plt.show()


