#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    :
2020-01-22 remove " and is_windows_os()" on if condition of plot_or_show function
"""
import typing
import itertools
import logging
import os
from datetime import datetime
from functools import lru_cache

import ffn
import matplotlib
# matplotlib.use('Qt5Agg')  # 需要 pip3 install PyQt5，windows 下無效
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as scs
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from ibats_utils.mess import date_2_str, is_windows_os, open_file_with_system_app

from ibats_common.backend.mess import get_cache_folder_path
from ibats_common.backend.label import calc_label2, calc_label3

logger = logging.getLogger(__name__)
logger.debug("matplotlib.backend => %s", matplotlib.get_backend())
register_matplotlib_converters()
FFN_VERSION = ffn.__version__
ALTER_BG_COLOR = '#e0e0e0'


def get_file_name(header, name=None):
    file_name = f'{header} {np.random.randint(10000)}.png' if name is None else f'{header} {name}.png'
    return file_name


def clean_cache():
    """
    清空cache目录
    :return:
    """
    folder_path = get_cache_folder_path()
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # print(root, dirs, files)
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def drawdown_plot(df: pd.DataFrame, perf_stats=None, col_name_list=None,
                  enable_show_plot=True, enable_save_plot=False, name=None, stg_run_id=None):
    """
    回撤曲线
    :param df:
    :param perf_stats:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :return:
    """
    if col_name_list is None:
        data_df = df.to_drawdown_series()
        col_name_list = list(df.columns)
    else:
        data_df = df[col_name_list].to_drawdown_series()

    if perf_stats is None:
        perf_stats = df.calc_stats()

    col_mdd_len_dic = {col_name: s.drawdown_details.Length.max()
                       for col_name, s in perf_stats.items() if s.drawdown_details is not None}
    col_mdd_dic = {col_name: mdd for col_name, mdd in data_df.min().items()}
    data_df.rename(
        columns={col_name: f"{col_name}[{mdd * 100:.2f}% {col_mdd_len_dic.setdefault(col_name, '')}]"
                 for col_name, mdd in col_mdd_dic.items()},
        inplace=True)

    ax = data_df.plot()
    ax.set_title(f"Drawdown {['{:.2f}%'.format(col_mdd_dic[_] * 100) for _ in col_name_list]}")

    file_name = get_file_name(f'drawdown', name=name)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                             file_name=file_name, stg_run_id=stg_run_id)

    return data_df, file_path


def plot_corr(df: pd.DataFrame, perf_stats=None,
              enable_show_plot=True, enable_save_plot=False, name=None, stg_run_id=None):
    """
    相关性矩阵图
    :param df:
    :param perf_stats:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :return:
    """
    if perf_stats is None:
        perf_stats = df.calc_stats()

    ax = perf_stats.plot_correlation()
    plt.suptitle("Correlation")

    file_name = get_file_name(f'correlation', name=name)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                             file_name=file_name, stg_run_id=stg_run_id)
    return file_path


def plot_rr_df(df: pd.DataFrame, col_name_list=None, enable_show_plot=True, enable_save_plot=False,
               name=None, stg_run_id=None):
    """
    Return Rate
    :param df:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :return:
    """
    if col_name_list is not None:
        data_df = df[col_name_list]
    else:
        data_df = df

    # """绘图函数"""
    ax = data_df.plot(grid=True)
    ax.set_title(
        f"Return Rate " if name is None else
        f"Return Rate [{name}] "
        f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    file_name = get_file_name(f'rr', name=name)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                             file_name=file_name, stg_run_id=stg_run_id)

    return file_path


def _test_plot_rr_df():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    file_path = plot_rr_df(df, col_name_list=['close'],
                           enable_show_plot=True, enable_save_plot=True, name='test_use', stg_run_id=2)
    logger.info(file_path)


def plot_scatter_matrix(df: pd.DataFrame, diagonal='hist', col_name_list=None, enable_show_plot=True,
                        enable_save_plot=False, name=None, stg_run_id=None):
    """
    plot scatter_matrix
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html
    :param df:
    :param diagonal: diagonal，必须且只能在{'hist', 'kde'}中选择1个，
    'hist'表示直方图(Histogram plot),'kde'表示核密度估计(Kernel Density Estimation)
    该参数是scatter_matrix函数的关键参数
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :return:
    """
    if col_name_list is not None:
        data_df = df[col_name_list]
    else:
        data_df = df

    # """绘图函数"""
    pd.plotting.scatter_matrix(data_df)
    plt.suptitle(
        f"Scatter Matrix " if name is None else
        f"Scatter Matrix [{name}] "
        f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    file_name = get_file_name('scatter_matrix', name=name)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                             file_name=file_name, stg_run_id=stg_run_id)
    return file_path


def hist_norm(data, bins=10, ax=None, enable_show_plot=True, enable_save_plot=False,
              name=None, stg_run_id=None, do_clr=True, folder_path=None, figsize=(6, 8)):
    """
    hist 分布图及正太分布曲线
    :param data:
    :param bins:
    :param ax:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :param do_clr:
    :param folder_path:
    :param figsize:
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

    # the histogram of the data
    _, _, patches = ax.hist(data, bins, density=True)
    n, bins_v, mean, std = plot_norm(
        data, bins=bins, ax=ax,
        enable_show_plot=False, enable_save_plot=False, do_clr=False)
    # ax.set_xlabel('pct change')
    # ax.set_ylabel('change rate')
    ax.set_title(f"{'Data' if name is None else name} Histogram (mean={mean:.4f} std={std:.4f})")

    file_name = get_file_name(f'hist', name=name)
    rr_plot_file_path = plot_or_show(
        enable_show_plot=enable_show_plot, enable_save_plot=enable_save_plot,
        file_name=file_name, stg_run_id=stg_run_id, do_clr=do_clr, folder_path=folder_path)
    return n, bins_v, rr_plot_file_path


def plot_norm(data: pd.Series, bins=10, ax=None, name=None,
              enable_save_plot=None, enable_show_plot=None, do_clr=None,
              folder_path=None, figsize=(6, 8)):
    """
    显示当前数据的正太分布曲线
    :param data:
    :param bins: bar 数量
    :param ax: 如果为None，则新建一个画布
    :param name: 图片 title 以及保存的文件名
    :param enable_save_plot:
    :param enable_show_plot:
    :param folder_path:
    :param figsize:
    :param do_clr:
    :return: n, bins_v, mean, std
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

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
    plt.grid(True)
    file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                             file_name=f'{name}.png', folder_path=folder_path)

    if enable_save_plot:
        return n, bins_v, mu, sigma, file_path
    else:
        return n, bins_v, mu, sigma


def hist_norm_sns(data, bins=10, ax=None, name=None,
                  enable_save_plot=None, enable_show_plot=None, do_clr=None,
                  folder_path=None, figsize=(6, 8)):
    """sns 方式 hist 分布图及正太分布曲线"""
    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

    sns.set_palette("hls")
    sns.distplot(
        data, bins=bins, ax=ax,
        fit=stats.norm,
        kde_kws={"color": "darkorange", "lw": 1, "label": "KDE", "linestyle": "--"},
        hist_kws={"color": "darkblue"})
    plt.grid(True)
    return plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                        file_name=f'{name}.png', folder_path=folder_path)


def wave_hist(df: pd.DataFrame, columns=None, bins=50, figure_4_each_col=True,
              col_transfer_dic: (dict, None) = None, enable_show_plot=True, enable_save_plot=False,
              name=None, stg_run_id=None):
    """
    波动率分布图
    :param df:
    :param perf_stats: df.calc_stats()
    :param columns: 显示哪些列
    :param bins: bar 数量
    :param figure_4_each_col: 每一个col单独一张图
    :param col_transfer_dic: 列值转换
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :param stg_run_id:
    :return:
    """
    if columns is not None:
        data_df = df.loc[:, [_ for _ in columns if _ in df.columns]].copy()
    else:
        data_df = df.copy()

    data_df.dropna(inplace=True)
    if col_transfer_dic is not None:
        for func, col_name_list in col_transfer_dic.items():
            if func == 'return':
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    if col_name not in data_df:
                        continue
                    rename_dic[col_name] = f"{col_name} {func}"
                    data_df.loc[:, col_name] = data_df[col_name].to_returns()
                else:
                    data_df.rename(columns=rename_dic, inplace=True)

            elif func == 'pct_change':
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    if col_name not in data_df:
                        continue
                    rename_dic[col_name] = f"{col_name} {func}"
                    data_df.loc[:, col_name] = data_df[col_name].pct_change()
                else:
                    data_df.rename(columns=rename_dic, inplace=True)
            else:
                rename_dic = {}
                for col_name in col_transfer_dic[func]:
                    if col_name not in data_df:
                        continue
                    rename_dic[col_name] = f"{col_name} {str(func)}"
                    data_df.loc[:, col_name] = func(data_df[col_name])
                else:
                    data_df.rename(columns=rename_dic, inplace=True)

        data_df.dropna(inplace=True)

    columns = list(data_df.columns)
    n_bins_dic, file_path = {}, None
    if figure_4_each_col:
        for col_name in columns:
            data = data_df[col_name]
            try:
                data = data[~np.isinf(data)].dropna()
            except TypeError as exp:
                logger.exception('column %s 数据类型无效，无法进行 np.isinf 计算', col_name)
                raise exp from exp

            n, bins_v, file_path_sub = hist_norm(
                data, bins=bins, enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                name=col_name, stg_run_id=stg_run_id)
            n_bins_dic[col_name] = (n, bins_v)
            if file_path is None:
                file_path = [file_path_sub]
            else:
                file_path.append(file_path_sub)

    else:

        ax = data_df.hist(bins=bins)
        if isinstance(ax, np.ndarray):
            # ax 总是一个 偶数 出现，因此，需要进行一次长度对齐
            ax_list = list(itertools.chain(*ax))[:len(columns)]
        else:
            ax_list = [ax]

        for col_name, ax_sub in zip(columns, ax_list):
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

        plt.grid(True)
        file_name = get_file_name(f'hist', name=name)
        file_path = plot_or_show(enable_show_plot=enable_show_plot, enable_save_plot=enable_save_plot,
                                 file_name=file_name, stg_run_id=stg_run_id)

    return n_bins_dic, file_path


def _test_wave_hist():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    n_bins_dic, file_path = wave_hist(df, figure_4_each_col=False, col_transfer_dic=col_transfer_dic,
                                      enable_save_plot=True, enable_show_plot=True, name="test_use", stg_run_id=1)


def hist_n_rr(df: pd.DataFrame, n_days, columns=None, bins=50,
              enable_show_plot=True, enable_save_plot=False, name=None, stg_run_id=None):
    """
    未来N日收益率分布图
    :param df:
    :param n_days: 计算未来 N 日的收益率最高值，最低值
    :param columns: 显示哪些列
    :param bins: bar 数量
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :return:
    """
    if columns is not None:
        data_df = df.loc[:, columns].dropna().copy()
    else:
        data_df = df.dropna().copy()

    column_name_list, df_dic, quantile_dic, n_bins_dic, file_path_dic = list(data_df.columns), {}, {}, {}, {}
    for n_day in n_days:
        # 计算各个列未来N日收益率波动
        max_df = data_df.rename(
            columns={_: _ + ' max rr' for _ in column_name_list}
        ).rolling(n_day).apply(
            lambda x: max(x / x[0]) - 1, raw=True).shift(-(n_day - 1))
        min_df = data_df.rename(
            columns={_: _ + ' min rr' for _ in column_name_list}
        ).rolling(n_day).apply(
            lambda x: min(x / x[0]) - 1, raw=True).shift(-(n_day - 1))
        # 合并数据
        merged_df = max_df.join(min_df).dropna()
        for col_name in column_name_list:
            col_names = [col_name + ' max rr', col_name + ' min rr']
            new_df = merged_df[col_names].copy()
            df_list = [new_df]
            # 计算收益率分位数信息
            idxmax_s = abs(new_df).idxmax(axis=1)
            # df.loc[idxmax_s != 'close max rr', 'close max rr'] = np.nan
            # df.loc[idxmax_s != 'close min rr', 'close min rr'] = np.nan
            label_quantile_dic = {}
            for label in new_df.columns:
                data_s = new_df.loc[idxmax_s == label, label].dropna()
                df_list.append(data_s)
                label_quantile_dic[label] = data_s.quantile([0.2, 0.33, 0.5, 0.66, 0.8])

            df_dic[(n_day, col_name)] = df_list
            quantile_dic[(n_day, col_name)] = pd.DataFrame(label_quantile_dic).T

    # 输出图片
    for (n_day, col_name), df_list in df_dic.items():
        hist_kwargs = dict(histtype='stepfilled', alpha=0.8, bins=bins)
        n_list, bins_v_list, axis_list, labels = [], [], [], []
        for data_s in df_list[1:]:
            n, bins_v, axis = plt.hist(data_s.dropna(), **hist_kwargs)
            labels.append(data_s.name)
            axis_list.extend(axis)

        plt.grid(True)
        plt.legend(axis_list, labels, loc='upper right')
        if name is None:
            title = f'hist of max/min {col_name} rr in {n_day} days'
        else:
            title = f'{name} hist of max/min {col_name} rr in {n_day} days'

        plt.suptitle(title)
        n_bins_dic[(n_day, col_name)] = (n_list, bins_v_list)

        file_name = get_file_name(f'hist {n_day} {col_name} future', name=name)
        file_path = plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
                                 file_name=file_name, stg_run_id=stg_run_id)
        if file_path is not None:
            file_path_dic[(n_day, col_name)] = file_path

    ret_dic = dict(df_dic=df_dic, n_bins_dic=n_bins_dic, quantile_dic=quantile_dic)
    return ret_dic, file_path_dic


def _test_hist_n_rr():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    ret_dic, file_path_dic = hist_n_rr(df, n_days=[3, 5], columns=['close'], enable_save_plot=True, stg_run_id=3)
    for k, v in ret_dic['quantile_dic'].items():
        logger.info('%s -> \n%s', k, v)


def label_distribution(close_df: pd.DataFrame, min_rr: float, max_rr: float, max_future: int,
                       name=None, **enable_kwargs):
    """
    输出分类标签在行情图中的分布情况
    :param close_df:
    :param min_rr:
    :param max_rr:
    :param max_future:为空则进行2分类，不为空则3分类
    :param enable_kwargs:
    :param name:
    :return:
    """
    logger.debug('%s [%f ~ %f] max_future=%d, name="%s"', close_df.shape, min_rr, max_rr, max_future, name)
    value_arr = close_df.to_numpy()
    if max_future is None:
        target_arr = calc_label2(value_arr, min_rr, max_rr, one_hot=False, dtype='int')
    else:
        target_arr = calc_label3(value_arr, min_rr, max_rr, max_future=max_future, one_hot=False, dtype='int')

    ax = close_df.plot(grid=True)
    plt.suptitle(f'label [{min_rr * 100:.2f}% ~ {max_rr * 100:.2f}%]')
    x_values = list(close_df.index)
    colors = [None, ALTER_BG_COLOR, '#d62728']

    value_count = len(value_arr)
    label_list = list(set(target_arr))
    distribution_rate_df = pd.DataFrame(
        {_: {'pct': sum(target_arr == _) / value_count} for _ in label_list}, columns=label_list)
    for label, range_from, range_to in get_range_num_iter(target_arr):
        color = colors[int(label)]
        if color is None:
            continue
        p = plt.axvspan(x_values[range_from], x_values[range_to], facecolor=color, alpha=0.5)

    file_name = get_file_name(f'label distribution', name=name)
    file_path = plot_or_show(file_name=file_name, **enable_kwargs)
    return distribution_rate_df, file_path


def show_dl_accuracy(real_ys, pred_ys, close_df: pd.DataFrame, split_point_list=None,
                     base_line_list: (None, dict) = None, show_moving_avg=False):
    """
    将实际标记与预测标记进行对比并结合行情close_df显示在plot图上
    :param real_ys:
    :param pred_ys:
    :param close_df:
    :param split_point_list:
    :param base_line_list:
    :param show_moving_avg: 展示滚动平均收益率，如果展示，则会显示 三张图，否则两张
    :return:
    """
    trade_date_index = close_df.index
    date_from_str, date_to_str = date_2_str(trade_date_index[0]), date_2_str(trade_date_index[-1])
    # 检查长度是否一致
    if not (len(real_ys) == len(pred_ys) == close_df.shape[0]) or len(real_ys) == 0:
        logger.error("[%s - %s] len(real_ys)=%d, len(pred_ys)=%d, close_df.shape[0]=%d 不一致",
                     date_from_str, date_to_str, len(real_ys), len(pred_ys), close_df.shape[0])
        return
    enable_kwargs = dict(enable_save_plot=False, enable_show_plot=False, do_clr=False)
    if show_moving_avg:
        fig = plt.figure(figsize=(12, 16))
        fig_idx = 310
    else:
        fig = plt.figure(figsize=(12, 14))
        fig_idx = 210

    # 分析成功率
    # 第一章图：累计平均成功率
    is_fit_arr = pred_ys == real_ys
    accuracy = sum(is_fit_arr) / len(pred_ys) * 100
    logger.info("模型准确率 [%s - %s] accuracy: %.2f%%", date_from_str, date_to_str, accuracy)
    accuracy_list, fit_sum = [], 0
    for tot_count, (is_fit, trade_date) in enumerate(zip(is_fit_arr, trade_date_index), start=1):
        if is_fit:
            fit_sum += 1
        accuracy_list.append(fit_sum / tot_count)

    accuracy_df = pd.DataFrame({'accuracy': accuracy_list}, index=trade_date_index)
    ax = fig.add_subplot(fig_idx + 1)
    plot_accuracy(accuracy_df, close_df, ax=ax, base_line_list=base_line_list,
                  name=f'Accumulation Avg Accuracy [{date_from_str}{date_to_str}]',
                  split_point_list=split_point_list, **enable_kwargs)

    # 第一章图：累计平均成功率
    real_pred_df = pd.DataFrame({'pred': pred_ys, 'real': real_ys}, index=trade_date_index)
    ax2 = fig.add_subplot(fig_idx + 2)
    plot_accuracy(accuracy_df, close_df, ax=ax2, base_line_list=base_line_list,
                  name=f'Accumulation Avg Accuracy [{date_from_str}{date_to_str}]',
                  real_pred_df=real_pred_df, **enable_kwargs)
    # 第三章图：移动平均成功率
    if show_moving_avg:
        accuracy_list, win_size = [], 30
        is_fit_df = pd.DataFrame({'fit': is_fit_arr}, index=trade_date_index)
        accuracy_df = is_fit_df.rolling(win_size, min_periods=1).apply(lambda x: sum(x) / len(x), raw=True)
        for idx in range(win_size, len(is_fit_arr)):
            accuracy_list.append(sum(is_fit_arr[idx - win_size:idx] / win_size))

        # close2_df = close_df.iloc[win_size:]
        # accuracy_df = pd.DataFrame({'accuracy': accuracy_list}, index=close2_df.index)
        ax3 = fig.add_subplot(fig_idx + 3)
        plot_accuracy(accuracy_df, close_df, ax=ax3, base_line_list=base_line_list,
                      name=f'{win_size} Moving Avg Accuracy [{date_from_str}{date_to_str}]',
                      split_point_list=split_point_list, **enable_kwargs)

    # 展示图片
    file_name = f"accuracy [{date_from_str}-{date_to_str}].png"
    file_path = plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=file_name)
    return file_path


def _test_show_dl_accuracy():
    real_ys, pred_ys = np.random.randint(1, 3, size=100), np.random.randint(1, 3, size=100)
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)

    split_point_list = np.random.randint(len(date_arr), size=10)
    split_point_list.sort()
    split_point_list = date_arr[split_point_list]
    base_line_list = [0.3, 0.6]

    show_dl_accuracy(real_ys, pred_ys, close_df, split_point_list, base_line_list, show_moving_avg=True)


def show_drl_accuracy(real_label_s, action_s, close_df: pd.DataFrame, split_point_list=None,
                      show_moving_avg=False):
    """
    将实际标记与预测标记进行对比并结合行情close_df显示在plot图上
    :param real_label_s:
    :param action_s:
    :param close_df:
    :param split_point_list:
    :param show_moving_avg: 展示滚动平均收益率，如果展示，则会显示 三张图，否则两张
    :return:
    """
    trade_date_index = close_df.index
    date_from_str, date_to_str = date_2_str(trade_date_index[0]), date_2_str(trade_date_index[-1])
    # 检查长度是否一致
    if not (len(real_label_s) == len(action_s) == close_df.shape[0]) or len(real_label_s) == 0:
        logger.error("[%s - %s] len(real_label_s)=%d, len(action_s)=%d, close_df.shape[0]=%d 不一致",
                     date_from_str, date_to_str, len(real_label_s), len(action_s), close_df.shape[0])
        return
    enable_kwargs = dict(enable_save_plot=False, enable_show_plot=False, do_clr=False)
    if show_moving_avg:
        fig = plt.figure(figsize=(12, 16))
        fig_idx = 310
    else:
        fig = plt.figure(figsize=(12, 14))
        fig_idx = 210

    # 分析成功率
    # 第一章图：累计平均成功率
    is_fit_arr = action_s == real_label_s
    accuracy = sum(is_fit_arr) / len(action_s) * 100
    logger.info("模型准确率 [%s - %s] accuracy: %.2f%%", date_from_str, date_to_str, accuracy)
    accuracy_list, fit_sum = [], 0
    for tot_count, (is_fit, trade_date) in enumerate(zip(is_fit_arr, trade_date_index), start=1):
        if is_fit:
            fit_sum += 1
        accuracy_list.append(fit_sum / tot_count)

    accuracy_df = pd.DataFrame({'accuracy': accuracy_list}, index=trade_date_index)
    ax = fig.add_subplot(fig_idx + 1)
    plot_accuracy(accuracy_df, close_df, ax=ax,
                  name=f'Accumulation Avg Accuracy [{date_from_str}{date_to_str}]',
                  split_point_list=split_point_list, **enable_kwargs)

    # 第一章图：累计平均成功率
    real_pred_df = pd.DataFrame({'pred': action_s, 'real': real_label_s}, index=trade_date_index)
    ax2 = fig.add_subplot(fig_idx + 2)
    plot_accuracy(accuracy_df, close_df, ax=ax2,
                  name=f'Accumulation Avg Accuracy [{date_from_str}{date_to_str}]',
                  real_pred_df=real_pred_df, **enable_kwargs)
    # 第三章图：移动平均成功率
    if show_moving_avg:
        accuracy_list, win_size = [], 30
        is_fit_df = pd.DataFrame({'fit': is_fit_arr}, index=trade_date_index)
        accuracy_df = is_fit_df.rolling(win_size, min_periods=1).apply(lambda x: sum(x) / len(x), raw=True)
        for idx in range(win_size, len(is_fit_arr)):
            accuracy_list.append(sum(is_fit_arr[idx - win_size:idx] / win_size))

        # close2_df = close_df.iloc[win_size:]
        # accuracy_df = pd.DataFrame({'accuracy': accuracy_list}, index=close2_df.index)
        ax3 = fig.add_subplot(fig_idx + 3)
        plot_accuracy(accuracy_df, close_df, ax=ax3,
                      name=f'{win_size} Moving Avg Accuracy [{date_from_str}{date_to_str}]',
                      split_point_list=split_point_list, **enable_kwargs)

    # 展示图片
    file_name = f"accuracy [{date_from_str}-{date_to_str}].png"
    file_path = plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=file_name)
    return file_path


def plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=None, stg_run_id=None, do_clr=True,
                 folder_path=None):
    """
    展示或保存图片
    :param enable_save_plot:
    :param enable_show_plot:
    :param file_name:
    :param stg_run_id:
    :param do_clr:
    :param folder_path:
    """
    if enable_save_plot:
        if folder_path is None:
            if stg_run_id is None:
                folder_path = get_cache_folder_path()
            else:
                folder_path = os.path.join(get_cache_folder_path(), str(stg_run_id))
        else:
            folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if file_name is None:
            file_name = 'output.png'
        if not file_name.endswith('.png'):
            file_name += '.png'
        file_path = os.path.join(folder_path, file_name)
        logger.debug("save to %s", file_path)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        # 2020-01-22 remove " and is_windows_os()" on if condition of plot_or_show function
        if file_path is not None:
            open_file_with_system_app(file_path)
        else:
            plt.show()

    if do_clr:
        plt.cla()
        plt.clf()
        plt.close()

    return file_path


def _test_label_distribution():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    distribution_rate_df, file_path = label_distribution(df['close'], min_rr=-0.01, max_rr=0.01, max_future=3,
                                                         enable_save_plot=True, enable_show_plot=True, stg_run_id=4)
    logger.info('\n%s', distribution_rate_df)
    logger.info(file_path)


def get_range_num_iter(arr):
    val_cur, range_from, range_to = arr[0], 0, None
    for range_to, val in enumerate(arr[1:], start=1):
        if val != val_cur:
            yield val_cur, range_from, range_to
            range_from = range_to
            val_cur = val
    else:
        range_to += 1
        yield val_cur, range_from, range_to


def _test_n_days_rr_distribution():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    ret_dic, file_path_dic, enable_kwargs = {}, {}, dict(enable_save_plot=True, enable_show_plot=True, stg_run_id=5)
    tmp_dic, file_path = hist_n_rr(df, n_days=[3, 5], columns=['close'], **enable_kwargs)
    ret_dic['hist_future_n_rr'] = tmp_dic
    file_path_dic['hist_future_n_rr'] = file_path

    from collections import defaultdict
    file_path_dic['label_distribution'] = defaultdict(dict)
    ret_dic['label_distribution'] = defaultdict(dict)
    for (n_day, col_name), quantile_df in tmp_dic['quantile_dic'].items():
        path_dic = file_path_dic['label_distribution'][(n_day, col_name)]
        distribution_dic = ret_dic['label_distribution'][(n_day, col_name)]
        col_count = quantile_df.shape[1]
        for n in range(col_count):
            max_rr = quantile_df.iloc[0, n]
            min_rr = quantile_df.iloc[1, col_count - n - 1]
            distribution_rate_df, file_path = label_distribution(
                df['close'], min_rr=min_rr, max_rr=max_rr, max_future=n_day,
                name=f"{col_name}[{min_rr * 100:.2f}%-{max_rr * 100:.2f}%]", **enable_kwargs)
            path_dic[(min_rr, max_rr)] = file_path
            distribution_dic[(min_rr, max_rr)] = distribution_rate_df


def plot_accuracy(accuracy_df, close_df, split_point_list=None, ax=None,
                  name=None, base_line_list: (None, dict) = None, real_pred_df=None, **enable_kwargs):
    """
    显示成功率与行情叠加走势，如果 real_pred_df is not None 则叠加预测与真实标记
    :param accuracy_df:
    :param close_df:
    :param split_point_list:
    :param ax:
    :param name:
    :param base_line_list:
    :param real_pred_df:
    :param enable_kwargs:
    :return:
    """
    if ax is None:
        fig = plt.figure()  #
        ax = fig.add_subplot(111)
        if "enable_save_plot" in enable_kwargs:
            enable_kwargs["enable_save_plot"] = False
        if "enable_show_plot" in enable_kwargs:
            enable_kwargs["enable_show_plot"] = False
        if "do_clr" in enable_kwargs:
            enable_kwargs["do_clr"] = False
    else:
        if "enable_save_plot" in enable_kwargs:
            enable_kwargs["enable_save_plot"] = True
        if "enable_show_plot" in enable_kwargs:
            enable_kwargs["enable_show_plot"] = True
        if "do_clr" in enable_kwargs:
            enable_kwargs["do_clr"] = True

    accuracy_df = accuracy_df.copy()
    l1 = ax.plot(accuracy_df, color='r', label='accuracy')
    if real_pred_df is not None:
        for x0, x1, data in get_data_range_iter(real_pred_df['pred'], extent_left=True):
            if data == 0:
                p = plt.axvspan(x0, x1, ymin=0.66, ymax=1, facecolor='cornflowerblue', alpha=0.5)
            elif data == 1:
                p = plt.axvspan(x0, x1, ymin=0.66, ymax=1, facecolor='lightcoral', alpha=0.5)
            elif data == 2:
                p = plt.axvspan(x0, x1, ymin=0.66, ymax=1, facecolor='lightseagreen', alpha=0.5)

        for x0, x1, data in get_data_range_iter(real_pred_df['real'], extent_left=True):
            if data == 0:
                p = plt.axvspan(x0, x1, ymin=0, ymax=0.33, facecolor='cornflowerblue', alpha=0.5)
            elif data == 1:
                p = plt.axvspan(x0, x1, ymin=0, ymax=0.33, facecolor='lightcoral', alpha=0.5)
            elif data == 2:
                p = plt.axvspan(x0, x1, ymin=0, ymax=0.33, facecolor='lightseagreen', alpha=0.5)

        is_fit_s = real_pred_df.apply(lambda x: x['real'] if x['real'] == x['pred'] else -2, axis=1)
        for x0, x1, data in get_data_range_iter(is_fit_s, extent_left=True):
            if data == 0:
                p = plt.axvspan(x0, x1, ymin=0.33, ymax=0.66, facecolor='cornflowerblue', alpha=0.5)
            elif data == 1:
                p = plt.axvspan(x0, x1, ymin=0.33, ymax=0.66, facecolor='lightcoral', alpha=0.5)
            elif data == 2:
                p = plt.axvspan(x0, x1, ymin=0.33, ymax=0.66, facecolor='lightseagreen', alpha=0.5)
            else:
                p = plt.axvspan(x0, x1, ymin=0.33, ymax=0.66, facecolor='red', alpha=0.5)

    ax2 = ax.twinx()
    l2 = ax2.plot(close_df, label='md')
    lns = l1 + l2
    if base_line_list is not None:
        acc = base_line_list[0]
        if acc is not None and acc != 0:
            accuracy_df['train_accuracy'] = acc
            lns += ax.plot(accuracy_df['train_accuracy'], linestyle='--', color='b', label='train_accuracy')
        acc = base_line_list[1]
        if acc is not None and acc != 0:
            accuracy_df['validation_accuracy'] = acc
            lns += ax.plot(accuracy_df['validation_accuracy'], linestyle='--', color='r', label='validation_accuracy')

    accuracy_df['50%'] = 0.5
    lns += ax.plot(accuracy_df['50%'], color='b', label='train_accuracy')

    plt.legend(lns, [_.get_label() for _ in lns], loc=0)
    plt.grid(True)
    if name is not None:
        # 解决中文字符无法显示问题，稍后将其范化
        # font = get_font_properties()
        # plt.suptitle(name, fontproperties=font)
        plt.title(name)

    # 分段着色
    if split_point_list is not None and len(split_point_list) > 2:
        x0, x1 = None, None
        for num, point in enumerate(split_point_list):
            if num % 2 == 0:
                x0 = point
            else:
                x1 = point
                if num >= 1:
                    p = plt.axvspan(x0, x1, facecolor=ALTER_BG_COLOR, alpha=0.5)

    datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')
    file_name = f"{datetime_str}" if name is None else f"{name}_{datetime_str}"
    file_path = plot_or_show(file_name=file_name, **enable_kwargs)
    return file_path


def plot_twin(df_list, df2, ax=None, name=None, enable_save_plot=None, enable_show_plot=None, do_clr=None,
              folder_path=None, y_scales_log=[False, False], in_sample_date_line=None, figsize=(6, 8)):
    """输出双坐标中图像"""
    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

    ax.set_prop_cycle(color=get_cmap('tab20').colors)
    if not isinstance(df_list, list):
        df_list = [df_list]

    l1, legend1, min_x, max_x = [], [], None, None
    for df, linestyle in zip(df_list,
                             ['-', ':', 'dashdotdotted', 'densely dashdotdotted', 'densely dotted'][:len(df_list)]):
        if df.shape[0] == 0:
            continue
        # 绘图
        if isinstance(df, pd.DataFrame):
            if df.shape[1] == 1:
                legend1 += [df.columns[0]]
            else:
                legend1 += list(df.columns)
        elif isinstance(df, pd.Series):
            legend1 += [df.name]
        else:
            legend1 += [""]

        min_x_tmp, max_x_tmp = min(df.index), max(df.index)
        if min_x is None or min_x > min_x_tmp:
            min_x = min_x_tmp
        if max_x is None or max_x > max_x_tmp:
            max_x = max_x_tmp

        l1 += ax.plot(df, linestyle=linestyle)
        if y_scales_log[0]:
            ax.set_yscale("log")

    # 分割着色
    if in_sample_date_line is not None:
        if min_x is None:
            logger.warning('min_x is None, cannot draw date_line')
        else:
            x1 = pd.to_datetime(in_sample_date_line)
            p = plt.axvspan(min_x, x1, facecolor=ALTER_BG_COLOR, alpha=0.3)

    if df2 is not None:
        ax2 = ax.twinx()
        ax2.set_prop_cycle(color=get_cmap('Set2').colors)
        if isinstance(df2, pd.DataFrame):
            if df2.shape[1] == 1:
                legend2 = [df2.columns[0]]
            else:
                legend2 = list(df2.columns)
        elif isinstance(df2, pd.Series):
            legend2 = [df2.name]
        else:
            legend2 = [""]
        l2 = ax2.plot(df2, linestyle='--')
        if y_scales_log[1]:
            ax2.set_yscale("log")
        # 设置 legend
        lns = l1 + l2
        legend = legend1 + legend2
    else:
        lns = l1
        legend = legend1

    # legend 参数设置
    # https://blog.csdn.net/helunqu2017/article/details/78641290
    # loc 0 best 3 lower left
    plt.legend(lns, legend, loc=3, frameon=False, ncol=2, fontsize='x-small')  # , framealpha=0.3
    plt.grid(True, color="gray", linewidth="0.5", alpha=0.3)  # , linestyle="-."
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    plt.xlim([min_x, max_x])  # x轴边界
    # 设置 title
    # plt.suptitle(name)
    plt.title(name)
    # 展示
    return plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                        file_name=f'{name}.png', folder_path=folder_path)


def _test_plot_twin():
    """测试 plot_twin"""
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)
    accuracy_df = pd.DataFrame({
        'acc1': np.cos(np.linspace(0, 10, 100)) + 9,
        'acc2': np.cos(np.linspace(1, 11, 100)) + 9
    }, index=date_index)
    accuracy2_df = pd.DataFrame({
        'acc_a': (np.cos(np.linspace(0, 10, 100)) + 2) * 10 - 5,
        'acc_b': np.cos(np.linspace(1, 11, 100)) + 9
    }, index=date_index)
    plot_twin([accuracy_df, accuracy2_df], close_df['close'], name='plot twin test log',
              y_scales_log=[True, False], in_sample_date_line='2018-02-01')


def plot_pair(df: pd.DataFrame, a_label, b_label, ax=None, name=None,
              enable_save_plot=None, enable_show_plot=None, do_clr=None,
              folder_path=None, figsize=(6, 8)):
    """画出配对交易连个品质的走势图"""
    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

    l1 = ax.plot(df[a_label], label=a_label, color='r')
    ax2 = ax.twinx()
    l2 = ax2.plot(df[b_label], label=b_label, color='b')
    lns = l1 + l2
    plt.legend(lns, [_.get_label() for _ in lns], loc=0)
    plt.grid(True)
    # 设置 title
    # plt.suptitle(name)
    plt.title(name)
    # 展示
    return plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                        file_name=f'{name}.png', folder_path=folder_path)


def _test_plot_pair():
    """测试 plot_twin"""
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)
    pair_df = pd.DataFrame({
        'instrument1': np.cos(np.linspace(0, 10, 100)) + 9,
        'instrument2': np.cos(np.linspace(1, 11, 100)) + 9
    }, index=date_index)
    plot_pair(pair_df, a_label='instrument1', b_label='instrument2', name='plot_pair_test', )


def plot_mean_std(s: typing.Union[pd.Series, np.ndarray], std_n=1, ax=None, name=None,
                  enable_save_plot=None, enable_show_plot=None, do_clr=None,
                  folder_path=None, figsize=(6, 8)):
    """画出走势图同时画出均线以及标准差线"""
    if not isinstance(s, pd.Series):
        s = pd.Series(s, name='value')

    if ax is None:
        fig = plt.figure(figsize=figsize)  #
        ax = fig.add_subplot(111)
        if enable_save_plot is None:
            enable_save_plot = False
        if enable_show_plot is None:
            enable_show_plot = False
        if do_clr is None:
            do_clr = False
    else:
        if enable_save_plot is None:
            enable_save_plot = True
        if enable_show_plot is None:
            enable_show_plot = True
        if do_clr is None:
            do_clr = True

    l1 = ax.plot(s, label=s.name, color='r')
    mean_value = s.mean()
    mean_s = pd.Series([mean_value for _ in range(s.shape[0])], index=s.index)
    std = s.std() * std_n
    l2 = ax.plot(mean_s, label='mean', color='k')
    l3 = ax.plot(mean_s + std, '--', label=f'mean+std*{std_n}', color='b')
    l4 = ax.plot(mean_s - std, '--', label=f'mean-std*{std_n}', color='b')
    lns = l1 + l2 + l3 + l4
    plt.legend(lns, [_.get_label() for _ in lns], loc=0)
    plt.grid(True)
    # 设置 title
    # plt.suptitle(name)
    plt.title(name)
    # 展示
    return plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                        file_name=f'{name}.png', folder_path=folder_path)


def _test_plot_mean_std():
    """测试 plot_twin"""
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)
    pair_df = pd.DataFrame({
        'instrument1': np.cos(np.linspace(0, 10, 100)) + 9,
        'instrument2': np.cos(np.linspace(1, 11, 100)) + 9
    }, index=date_index)
    gap_s = pair_df['instrument2'] - pair_df['instrument1']
    plot_mean_std(gap_s, name='plot_mean_std_test', )


def pair_plots(s1: pd.Series, s2: pd.Series, z_score=True,
               enable_save_plot=True, enable_show_plot=True, do_clr=True,
               folder_path=None, figsize=(6, 8)):
    """
    将 s1, s2 配对交易图显示
    """
    s1_name = s1.name
    s2_name = s2.name
    merged_df = pd.merge(s1, s2, left_index=True, right_index=True)
    logger.info("%s shape %s, %s shape %s, 合并后 %s",
                s1_name, s1.shape, s2_name, s2.shape, merged_df.shape)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :])
    ax3 = plt.subplot(gs[2, :])
    # 合并走势图
    plot_pair(
        merged_df, s1_name, s2_name,
        ax=ax1, name=f'{s1_name} & {s2_name}',
        enable_save_plot=False, enable_show_plot=False, do_clr=False)
    # 基差走势图
    gap_s = merged_df[s2_name] - merged_df[s1_name]
    if z_score:
        from scipy.stats import zscore
        gap_s = pd.Series(zscore(gap_s), index=merged_df.index, name=f'{s2_name} - {s1_name}')
    else:
        gap_s.name = f'{s2_name} - {s1_name}'

    plot_mean_std(
        gap_s,
        ax=ax2, name=f'{s2_name} - {s1_name}',
        enable_save_plot=False, enable_show_plot=False, do_clr=False)

    # 分布图
    hist_norm(
        gap_s, bins=50,
        ax=ax3, name=f'{s2_name} - {s1_name}{" by z_score" if z_score else ""}',
        enable_save_plot=False, enable_show_plot=False, do_clr=False
    )
    name = 'Pair Plots'
    return plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, do_clr=do_clr,
                        file_name=f'{name}.png', folder_path=folder_path)


def _test_pair_plot():
    """测试 plot_twin"""
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)
    pair_df = pd.DataFrame({
        'instrument1': np.cos(np.linspace(0, 10, 100)) + 9,
        'instrument2': np.cos(np.linspace(1, 11, 100)) + 9
    }, index=date_index)
    pair_plots(pair_df['instrument1'], pair_df['instrument2'])


@lru_cache()
def get_font_properties():
    is_win = is_windows_os()
    if is_win:
        # 调用系统字体  C:\WINDOWS\Fonts
        font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\FZSTK.TTF", size=14)
    else:
        # 可以通过 from ibats_utils.mess import get_chinese_font_iter 获取系统有效的中文字体
        # 然后在 /usr/share/fonts/truetype 中查找，
        # 例如：Droid Sans Fallback 字体，对应路径 droid/DroidSansFallbackFull.ttf
        font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size=14)

    return font


def _test_plot_accuracy(single_figure=True):
    """测试 plot_accuracy"""
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=99))
    date_index = pd.DatetimeIndex(date_arr)
    close_df = pd.DataFrame({'close': np.sin(np.linspace(0, 10, 100))}, index=date_index)
    accuracy_df = pd.DataFrame({'acc': np.cos(np.linspace(0, 10, 100))}, index=date_index)
    real_pred_df = pd.DataFrame({
        'pred': close_df.apply(lambda x: 1 if x.iloc[0] > 0.3 else 2 if x.iloc[0] < -0.3 else 0, axis=1),
        'real': accuracy_df.apply(lambda x: 1 if x.iloc[0] > 0.3 else 2 if x.iloc[0] < -0.3 else 0, axis=1),
    })
    split_point_list = np.random.randint(len(date_arr), size=10)
    split_point_list.sort()
    split_point_list = date_arr[split_point_list]
    base_line_list = [0.3, 0.6]
    if single_figure:
        enable_kwargs = dict(enable_save_plot=False, enable_show_plot=False, do_clr=False, stg_run_id=6)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(211)
        plot_accuracy(accuracy_df, close_df, ax=ax, name='测试使用', split_point_list=split_point_list,
                      base_line_list=base_line_list, **enable_kwargs)
        ax = fig.add_subplot(212)
        plot_accuracy(accuracy_df, close_df, ax=ax, name='测试使用',
                      base_line_list=base_line_list, real_pred_df=real_pred_df, **enable_kwargs)
        plot_or_show(enable_show_plot=True, enable_save_plot=True, file_name='测试使用all in one', stg_run_id=6)
    else:
        plot_accuracy(accuracy_df, close_df, name='测试使用2', split_point_list=split_point_list,
                      base_line_list=base_line_list, stg_run_id=6)


def get_data_range_iter(s: pd.Series, extent_left=False):
    """
    从序列数据中迭代输出每一段相同数据的index范围
    :param s:
    :param extent_left: 左边界与上一个迭代的右边界使用同一个值
    :return:
    """
    is_new_range, idx_from, idx_to, data = True, s.index[0], None, None
    for (idx_to, data), (_, d2) in zip(s.items(), s.shift(-1).items()):
        if is_new_range and not extent_left:
            idx_from = idx_to
            is_new_range = False
        if data != d2 and not (np.isnan(data) and np.isnan(d2)):
            yield idx_from, idx_to, data
            if extent_left:
                idx_from = idx_to
            is_new_range = True
    else:
        if not is_new_range:
            yield idx_from, idx_to, data


def ts_plot(y, lags=None, figsize=(10, 12), style='bmh', dropna=True, drop_n_std=None):
    """
    对时间序列数据进行ACF PACF展示
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    if dropna:
        y = y.dropna()

    if drop_n_std is not None:
        y_std_n = np.std(y) * drop_n_std
        y = y[np.abs(y) < y_std_n]

    with plt.style.context(style):  # 定义局部样式
        fig = plt.figure(figsize=figsize)
        layout = (5, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        hist_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        pacf_ax = plt.subplot2grid(layout, (3, 0), colspan=2)
        qq_ax = plt.subplot2grid(layout, (4, 0))
        pp_ax = plt.subplot2grid(layout, (4, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots[ACF->q, PACF->p]')
        y.plot(ax=hist_ax, kind='hist', bins=25)
        plot_acf(y, lags=lags, ax=acf_ax)  # 自相关系数ACF图 , alpha=0.5
        plot_pacf(y, lags=lags, ax=pacf_ax)  # 偏相关系数PACF图 , alpha=0.5
        sns.despine()
        sm.qqplot(y, line='s', ax=qq_ax)  # QQ图检验是否是正太分布
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()


if __name__ == "__main__":
    # _test_plot_rr_df()
    # _test_wave_hist()
    # _test_hist_n_rr()
    # _test_label_distribution()
    # _test_n_days_rr_distribution()
    # _test_plot_accuracy()
    # _test_show_dl_accuracy()
    # _test_plot_twin()
    # _test_plot_pair()
    # _test_plot_mean_std()
    _test_pair_plot()
