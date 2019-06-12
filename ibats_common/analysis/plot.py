#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import itertools
import logging
import os
from datetime import datetime
from functools import lru_cache

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from pylab import mpl
from pandas.plotting import register_matplotlib_converters

import numpy as np
import pandas as pd
import seaborn as sns
from ibats_utils.mess import date_2_str, is_windows_os
from scipy import stats

from ibats_common.analysis import get_cache_folder_path
from ibats_common.backend.label import calc_label2, calc_label3

logger = logging.getLogger(__name__)
register_matplotlib_converters()


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
                  enable_show_plot=True, enable_save_plot=False, name=None):
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

    if enable_save_plot:
        file_name = get_file_name(f'drawdown', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()

    plt.cla()
    plt.clf()

    return data_df, file_path


def plot_corr(df: pd.DataFrame, perf_stats=None,
              enable_show_plot=True, enable_save_plot=False, name=None):
    """
    相关性矩阵图
    :param df:
    :param perf_stats:
    :param col_name_list:
    :param enable_show_plot:
    :param enable_save_plot:
    :param name:
    :return:
    """
    if perf_stats is None:
        perf_stats = df.calc_stats()

    ax = perf_stats.plot_correlation()
    plt.suptitle("Correlation")

    if enable_save_plot:
        file_name = get_file_name(f'correlation', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()

    plt.cla()
    plt.clf()

    return file_path


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

    # """绘图函数"""
    ax = data_df.plot(grid=True)
    ax.set_title(
        f"Return Rate " if name is None else f"Return Rate [{name}] "
        f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    if enable_save_plot:
        file_name = get_file_name(f'rr', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()

    plt.cla()
    plt.clf()

    return file_path


def _test_plot_rr_df():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    file_path = plot_rr_df(df, col_name_list=['close'],
                           enable_show_plot=True, enable_save_plot=True)
    logger.info(file_path)


def plot_scatter_matrix(df: pd.DataFrame, diagonal='hist', col_name_list=None, enable_show_plot=True,
                        enable_save_plot=False, name=None):
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
    :return:
    """
    if col_name_list is not None:
        data_df = df[col_name_list]
    else:
        data_df = df

    # """绘图函数"""
    pd.plotting.scatter_matrix(data_df)
    plt.suptitle(
        f"Scatter Matrix " if name is None else f"Scatter Matrix [{name}] "
        f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    if enable_save_plot:
        file_name = get_file_name('scatter_matrix', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()

    plt.cla()
    plt.clf()

    return file_path


def hist_norm(data, bins=10, enable_show_plot=True, enable_save_plot=False, name=None):
    """hist 分布图及正太分布曲线"""
    n, bins_v, file_name = None, None, None

    # """绘图函数"""
    # ax = pct_change_s.hist(bins=50, density=1)
    fig, ax = plt.subplots()
    # the histogram of the data
    _, _, patches = ax.hist(data, bins, density=True)
    n, bins_v, mean, std = plot_norm(data, bins=bins, ax=ax)
    # ax.set_xlabel('pct change')
    # ax.set_ylabel('change rate')
    ax.set_title(f"{'Data' if name is None else name} Histogram (mean={mean:.4f} std={std:.4f})")

    file_name = get_file_name(f'hist', name=name)
    rr_plot_file_path = plot_or_show(enable_show_plot=enable_show_plot, enable_save_plot=enable_save_plot, file_name=file_name)

    return n, bins_v, rr_plot_file_path


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
    plt.grid(True)
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
    plt.grid(True)
    plt.show()


def wave_hist(df: pd.DataFrame, columns=None, bins=50, figure_4_each_col=True,
              col_transfer_dic: (dict, None) = None, enable_show_plot=True, enable_save_plot=False, name=None):
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
                data, bins=bins, enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, name=col_name)
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
        if enable_save_plot:
            file_name = get_file_name(f'hist', name=name)
            file_path = os.path.join(get_cache_folder_path(), file_name)
            plt.savefig(file_path, dpi=75)

        if enable_show_plot:
            plt.show()

        plt.cla()
        plt.clf()

    return n_bins_dic, file_path


def _test_wave_hist():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    n_bins_dic, file_path = wave_hist(df, figure_4_each_col=False, col_transfer_dic=col_transfer_dic)


def hist_n_rr(df: pd.DataFrame, n_days, columns=None, bins=50,
              enable_show_plot=True, enable_save_plot=False, name=None):
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

        if enable_save_plot:
            file_name = get_file_name(f'hist {n_day} {col_name} future', name=name)
            file_path = os.path.join(get_cache_folder_path(), file_name)
            plt.savefig(file_path, dpi=75)
            file_path_dic[(n_day, col_name)] = file_path

        if enable_show_plot:
            plt.show()

        plt.cla()
        plt.clf()

    ret_dic = dict(df_dic=df_dic, n_bins_dic=n_bins_dic, quantile_dic=quantile_dic)
    return ret_dic, file_path_dic


def _test_hist_n_rr():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    ret_dic, file_path_dic = hist_n_rr(df, n_days=[3, 5], columns=['close'], enable_save_plot=True)
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
    colors = [None, '#2ca02c', '#d62728']

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


def plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=None):
    if enable_save_plot:
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    if enable_show_plot:
        plt.show()
        # plt.cla()
        # plt.clf()

    return file_path


def _test_label_distribution():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    distribution_rate_df, file_path = label_distribution(df['close'], min_rr=-0.01, max_rr=0.01, max_future=3)
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
    ret_dic, file_path_dic, enable_kwargs = {}, {}, dict(enable_save_plot=True, enable_show_plot=True)
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
                  enable_save_plot=True, enable_show_plot=True, name=None, base_line_list: (None, dict) = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    l1 = ax.plot(accuracy_df, color='r', label='accuracy')

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
                    p = plt.axvspan(x0, x1, facecolor='#2ca02c', alpha=0.5)

    datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')
    file_name = f"{datetime_str}" if name is None else f"{name}_{datetime_str}"
    plot_or_show(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, file_name=file_name)


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
    split_point_list = np.random.randint(len(date_arr), size=10)
    split_point_list.sort()
    split_point_list = date_arr[split_point_list]
    base_line_list = [0.3, 0.6]
    if single_figure:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(211)
        plot_accuracy(accuracy_df, close_df, ax=ax, name='测试使用', split_point_list=split_point_list,
                      base_line_list=base_line_list, enable_save_plot=False, enable_show_plot=False)
        ax = fig.add_subplot(212)
        plot_accuracy(accuracy_df, close_df, ax=ax, name='测试使用2', split_point_list=split_point_list,
                      base_line_list=base_line_list, enable_save_plot=False, enable_show_plot=False)
        plot_or_show(enable_show_plot=True, enable_save_plot=True, file_name='测试使用all in one')
    else:
        plot_accuracy(accuracy_df, close_df, name='测试使用2', split_point_list=split_point_list,
                      base_line_list=base_line_list)


if __name__ == "__main__":
    # _test_plot_rr_df()
    # _test_wave_hist()
    # _test_hist_n_rr()
    # _test_label_distribution()
    # _test_n_days_rr_distribution()
    _test_plot_accuracy()
