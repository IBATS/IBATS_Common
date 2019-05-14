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
import itertools
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
    else:
        data_df = df[col_name_list].to_drawdown_series()

    if perf_stats is None:
        perf_stats = df.calc_stats()

    col_mdd_len_dic = {col_name: s.drawdown_details.Length.max()
                       for col_name, s in perf_stats.items() if s.drawdown_details is not None}
    col_mdd_dic = {col_name: mdd for col_name, mdd in data_df.min().items()}
    data_df.rename(
        columns={col_name: f"{col_name}[{mdd*100:.2f}% {col_mdd_len_dic.setdefault(col_name, '')}]"
                 for col_name, mdd in col_mdd_dic.items()},
        inplace=True)

    if enable_show_plot:
        ax = data_df.plot()
        ax.set_title(f"Drawdown {['{:.2f}%'.format(col_mdd_dic[_] * 100) for _ in df.columns]}")
        plt.show()

    if enable_save_plot:
        ax = data_df.plot()
        ax.set_title(f"Drawdown {['{:.2f}%'.format(col_mdd_dic[_] * 100) for _ in df.columns]}")
        file_name = get_file_name(f'drawdown', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

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

    def func():
        ax = perf_stats.plot_correlation()
        plt.suptitle("Correlation")

    if enable_show_plot:
        func()
        plt.show()

    if enable_save_plot:
        func()
        file_name = get_file_name(f'correlation', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

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

    def func():
        """绘图函数"""
        ax = data_df.plot()
        ax.set_title(
            f"Return Rate " if name is None else f"Return Rate [{name}] "  
            f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    if enable_show_plot:
        func()
        plt.show()

    if enable_save_plot:
        func()
        file_name = get_file_name(f'rr', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    return file_path


def plot_scatter_matrix(df: pd.DataFrame, diagonal='hist', col_name_list=None, enable_show_plot=True, enable_save_plot=False, name=None):
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

    def func():
        """绘图函数"""
        pd.plotting.scatter_matrix(data_df)
        plt.suptitle(
            f"Scatter Matrix " if name is None else f"Scatter Matrix [{name}] "  
            f"{date_2_str(min(data_df.index))} - {date_2_str(max(data_df.index))} ({data_df.shape[0]} days)")

    if enable_show_plot:
        func()
        plt.show()

    if enable_save_plot:
        func()
        file_name = get_file_name('scatter_matrix', name=name)
        file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(file_path, dpi=75)
    else:
        file_path = None

    return file_path


def hist_norm(data, bins=10, enable_show_plot=True, enable_save_plot=False, name=None):
    """hist 分布图及正太分布曲线"""
    n, bins_v, file_name = None, None, None

    def func():
        """绘图函数"""
        # ax = pct_change_s.hist(bins=50, density=1)
        fig, ax = plt.subplots()
        # the histogram of the data
        _, _, patches = ax.hist(data, bins, density=True)
        n, bins_v, mean, std = plot_norm(data, bins=bins, ax=ax)
        # ax.set_xlabel('pct change')
        # ax.set_ylabel('change rate')
        ax.set_title(f"{'Data' if name is None else name} Histogram (mean={mean:.4f} std={std:.4f})")

    if enable_show_plot:
        func()
        plt.show()

    if enable_save_plot:
        func()
        file_name = get_file_name(f'hist', name=name)
        rr_plot_file_path = os.path.join(get_cache_folder_path(), file_name)
        plt.savefig(rr_plot_file_path, dpi=75)

    return n, bins_v, file_name


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

        def func():
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

        if enable_show_plot:
            func()
            plt.show()

        if enable_save_plot:
            file_name = get_file_name(f'hist', name=name)
            file_path = os.path.join(get_cache_folder_path(), file_name)
            plt.savefig(file_path, dpi=75)

    return n_bins_dic, file_path


def _test_wave_hist():
    from ibats_common.example.data import load_data
    df = load_data('RB.csv').set_index('trade_date').drop('instrument_type', axis=1)
    df.index = pd.DatetimeIndex(df.index)
    col_transfer_dic = {
        'return': ['open', 'high', 'low', 'close', 'volume']
    }
    n_bins_dic, file_path = wave_hist(df, figure_4_each_col=False, col_transfer_dic=col_transfer_dic)


if __name__ == "__main__":
    _test_wave_hist()
