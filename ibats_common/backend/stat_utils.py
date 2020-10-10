"""
@author  : MG
@Time    : 2020/9/11 13:49
@File    : stat_utils.py
@contact : mmmaaaggg@163.com
@desc    : 用于提供各种统计学工具
"""
from typing import Union
import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import anderson, kstest

logger = logging.getLogger()
LABEL_T_SMALLER_THAN = 't-statistic Smaller Than'
LABEL_STATISTIC_LARGER_THAN = 'statistic larger Than'
LABEL_P_SIGNIFICANT = 'p-value significant'
LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS = 'rejection of original hypothesis'


def ks_test(x, cdf='norm', enable_save_plot=True, enable_show_plot=True,
            file_name="hist.png", do_clr=True, folder_path=None):
    """
    当前函数通过两种方法验证是否符合某一分布
    1）通过 kstest cdf 原假设是 服从 cdf 分布，
    2）anderson
    当 LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS 值 等于 1是，当前数据不符合给定分布
    """
    statistic, p_value = kstest(x, cdf=cdf)
    statistic, critical_values, significance_level = anderson(x, dist=cdf)
    output_s = pd.Series(
        [statistic, p_value],
        index=["KS test statistic", 'p-value'])
    output_s[LABEL_P_SIGNIFICANT] = 1 if p_value < 0.05 else 0
    output_s[LABEL_P_SIGNIFICANT] = 1 if p_value < 0.05 else 0
    output_s[LABEL_STATISTIC_LARGER_THAN] = None
    for n, (critical_value, level) in enumerate(zip(critical_values, significance_level)):
        key = f"{int(level)}%"
        critical_value = critical_values[n]
        output_s['Critical Value (%s)' % key] = critical_value
        # 由于显著水平是递减的 significance_level=array([ 15. ,  10. ,   5. ,   2.5,   1. ])，
        # 所以无需判断 output_s[LABEL_T_SMALLER_THAN] is None
        if statistic > critical_value:
            output_s[LABEL_STATISTIC_LARGER_THAN] = level / 100

    # 关于统计值与评价值的对比：当统计值大于这些评价值时，表示在对应的显著性水平下，原假设被拒绝，即不属于某分布。
    # 如果p-value 显著，且 statistic 大于 1% level对应的 critical_value 则极其显著拒绝原假设
    # output_s[LABEL_T_SMALLER_THAN] <= 0.01 其中 0.01 对应 level / 100 中 level = 1 的那一项，
    # 当 LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS 值 等于 1是，当前数据不符合给定分布
    output_s[LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS] = 1 if \
        output_s[LABEL_STATISTIC_LARGER_THAN] is not None \
        and output_s[LABEL_STATISTIC_LARGER_THAN] <= 0.01 \
        and output_s[LABEL_P_SIGNIFICANT] == 1 else 0

    if enable_save_plot or enable_show_plot:
        from ibats_common.analysis.plot import hist_norm
        hist_norm(x, enable_show_plot=enable_show_plot, enable_save_plot=enable_save_plot,
                  name=file_name, do_clr=do_clr, folder_path=folder_path)

    return output_s


def corr(df: pd.DataFrame, method="pearson", enable_save_plot=True, enable_show_plot=True,
         file_name="Corr.png", do_clr=True, folder_path=None):
    """
    相关性分析，并plot出图片
    """
    from ibats_common.analysis.plot import plot_or_show
    data = df.corr(method=method)
    if enable_save_plot or enable_save_plot:
        pd.plotting.scatter_matrix(data, figsize=[_ * 1.5 for _ in data.shape],
                                   c='k',
                                   marker='+',
                                   diagonal='hist',
                                   alpha=0.8,
                                   range_padding=0.0)
        plot_or_show(
            enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot,
            file_name=file_name, do_clr=do_clr, folder_path=folder_path)
    return data


def coint_test(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    协整检测
    如果 LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS 为 1 则代表机器显著的拒绝原假设，两序列存在协整关系
    """
    result = t_value, p_value, critical_values = coint(x, y)
    output_s = pd.Series(
        result[0:2],
        index=['Test Statistic', 'p-value'])
    output_s[LABEL_P_SIGNIFICANT] = 1 if p_value < 0.05 else 0
    output_s[LABEL_T_SMALLER_THAN] = None
    for n, value in enumerate([0.01, 0.05, 0.1]):
        key = f"{int(value * 100)}%"
        critical_value = critical_values[n]
        output_s['Critical Value (%s)' % key] = critical_value
        if output_s[LABEL_T_SMALLER_THAN] is None and t_value < critical_value:
            output_s[LABEL_T_SMALLER_THAN] = value

    # 如果p-value 显著，且 t-value小于1%则极其显著拒绝原假设
    output_s[LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS] = 1 if \
        output_s[LABEL_T_SMALLER_THAN] is not None \
        and output_s[LABEL_T_SMALLER_THAN] <= 0.01 \
        and output_s[LABEL_P_SIGNIFICANT] == 1 else 0
    return output_s


def adf_test(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.Series, pd.DataFrame]:
    """
    ADF测试，验证序列是否平稳，
    如果 LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS 为 1 则代表机器显著的拒绝原假设，该序列为平稳
    """
    if isinstance(data, pd.DataFrame):
        result = {key: adf_test(_) for key, _ in data.items()}
        output_df = pd.DataFrame(result)
        return output_df
    else:
        result = adfuller(data)
        t_value = result[0]
        p_value = result[1]
        output_s = pd.Series(
            result[0:4],
            index=['Test Statistic', 'p-value',
                   '#Lags Used', 'Number of Observations Used'])
        output_s[LABEL_P_SIGNIFICANT] = 1 if p_value < 0.05 else 0
        output_s[LABEL_T_SMALLER_THAN] = None
        for value in [0.01, 0.05, 0.1]:
            key = f"{int(value * 100)}%"
            critical_value = result[4][key]
            output_s['Critical Value (%s)' % key] = critical_value
            if output_s[LABEL_T_SMALLER_THAN] is None and t_value < critical_value:
                output_s[LABEL_T_SMALLER_THAN] = value

        # 如果p-value 显著，且 t-value小于1%则极其显著拒绝原假设
        output_s[LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS] = 1 if \
            output_s[LABEL_T_SMALLER_THAN] is not None \
            and output_s[LABEL_T_SMALLER_THAN] <= 0.01 \
            and output_s[LABEL_P_SIGNIFICANT] == 1 else 0
        return output_s


def adf_coint_test(x, y, max_diff=4):
    """
    集成同阶单整检测，与协整检测。数据只有符合“同阶单整”的情况下才会进行协整检测，否则返回 None
    :param x:
    :param y:
    :param max_diff:
    :return:
    """
    x_name = x.name if isinstance(x, pd.Series) else 'x'
    y_name = y.name if isinstance(y, pd.Series) else 'y'
    xy_df = pd.DataFrame({
        x_name: x,
        y_name: y,
    })
    diff_n = 0
    for diff_n in range(max_diff + 1):
        if diff_n > 0:
            data = xy_df.diff(diff_n)
        else:
            data = xy_df

        output_df = adf_test(data.dropna())
        logger.debug("ADF Test:\n%s", output_df)
        is_ok = output_df.loc[LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS, :] == 1
        if np.all(is_ok):
            logger.debug("当前数据 %s %s 同时符合 %d阶单整，即：I(%d)", x_name, y_name, diff_n, diff_n)
            break
    else:
        logger.warning("当前数据 %s %s 不存在 同阶单整", x_name, y_name)
        return None

    if diff_n > 0:
        x_diff = np.diff(x, diff_n)
        y_diff = np.diff(y, diff_n)
    else:
        x_diff = x
        y_diff = y

    return coint_test(x_diff, y_diff)


def _test_adf_test():
    from ibats_common.example.data import load_data
    folder_path = r'd:\github\IBATS_Common\ibats_common\example\data'
    df = load_data("RB.csv", folder_path=folder_path,
                   index_col=[0], parse_index_to_datetime=True)
    del df['instrument_type']
    # 结果类似如下
    #                                        open       high        low      close     volume         oi     amount
    # Test Statistic                      -1.2338    -1.1714    -1.1883    -1.1448    -4.0667    -3.1700    -4.0262
    # p-value                              0.6589     0.6858     0.6786     0.6969     0.0011     0.0218     0.0013
    # #Lags Used                           4.0000     1.0000     3.0000     0.0000    26.0000    11.0000    14.0000
    # Number of Observations Used      2,465.0000 2,468.0000 2,466.0000 2,469.0000 2,443.0000 2,458.0000 2,455.0000
    # p-value significant                  0.0000     0.0000     0.0000     0.0000     1.0000     1.0000     1.0000
    # t-value Smaller Than                   None       None       None       None     0.0100     0.0500     0.0100
    # Critical Value (1%)                 -3.4330    -3.4330    -3.4330    -3.4330    -3.4330    -3.4330    -3.4330
    # Critical Value (5%)                 -2.8627    -2.8627    -2.8627    -2.8627    -2.8627    -2.8627    -2.8627
    # Critical Value (10%)                -2.5674    -2.5674    -2.5674    -2.5674    -2.5674    -2.5674    -2.5674
    # rejection of original hypothesis          0          0          0          0          1          0          1
    print(adf_test(df))
    # 结果类似如下
    #                                        open       high        low      close     volume         oi     amount
    # Test Statistic                      -9.4852    -9.0850   -12.1689    -9.6170   -18.4331   -17.1111   -19.2670
    # p-value                              0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000
    # #Lags Used                          26.0000    23.0000    15.0000    26.0000    27.0000    19.0000    27.0000
    # Number of Observations Used      2,442.0000 2,445.0000 2,453.0000 2,442.0000 2,441.0000 2,449.0000 2,441.0000
    # p-value significant                  1.0000     1.0000     1.0000     1.0000     1.0000     1.0000     1.0000
    # t-value Smaller Than                 0.0100     0.0100     0.0100     0.0100     0.0100     0.0100     0.0100
    # Critical Value (1%)                 -3.4330    -3.4330    -3.4330    -3.4330    -3.4330    -3.4330    -3.4330
    # Critical Value (5%)                 -2.8627    -2.8627    -2.8627    -2.8627    -2.8627    -2.8627    -2.8627
    # Critical Value (10%)                -2.5674    -2.5674    -2.5674    -2.5674    -2.5674    -2.5674    -2.5674
    # rejection of original hypothesis          1          1          1          1          1          1          1
    print(adf_test((df.pct_change() / df).dropna()))


def _test_coint_test():
    from ibats_common.example.data import load_data
    folder_path = r'd:\github\IBATS_Common\ibats_common\example\data'
    df = load_data("RB.csv", folder_path=folder_path,
                   index_col=[0], parse_index_to_datetime=True)
    del df['instrument_type']
    # 结果类似如下
    # Test Statistic                     -23.7895
    # p-value                              0.0000
    # p-value significant                  1.0000
    # t-statistic Smaller Than             0.0100
    # Critical Value (1%)                 -3.9009
    # Critical Value (5%)                 -3.3386
    # Critical Value (10%)                -3.0462
    # rejection of original hypothesis          1
    # dtype: object
    print(coint_test(df['close'], df['open']))


def _test_adf_coint_test():
    from ibats_common.example.data import load_data
    folder_path = r'c:\github\IBATS_Common\ibats_common\example\data'
    df = load_data("RB.csv", folder_path=folder_path,
                   index_col=[0], parse_index_to_datetime=True)
    del df['instrument_type']
    # 结果类似如下
    # Test Statistic                     -23.7895
    # p-value                              0.0000
    # p-value significant                  1.0000
    # t-statistic Smaller Than             0.0100
    # Critical Value (1%)                 -3.9009
    # Critical Value (5%)                 -3.3386
    # Critical Value (10%)                -3.0462
    # rejection of original hypothesis          1
    # dtype: object
    print(adf_coint_test(df['close'], df['open']))


def _test_corr():
    from ibats_common.example.data import load_data
    folder_path = r'd:\github\IBATS_Common\ibats_common\example\data'
    df = load_data("RB.csv", folder_path=folder_path,
                   index_col=[0], parse_index_to_datetime=True)
    del df['instrument_type']
    corr(df)


def _test_ks_test():
    from ibats_common.example.data import load_data
    folder_path = r'c:\github\IBATS_Common\ibats_common\example\data'
    df = load_data("RB.csv", folder_path=folder_path,
                   index_col=[0], parse_index_to_datetime=True)
    del df['instrument_type']
    # 结果类似如下
    # KS test statistic                  15.1484
    # p-value                             0.0000
    # p-value significant                 1.0000
    # statistic larger Than               0.0100
    # Critical Value (15%)                0.5750
    # Critical Value (10%)                0.6550
    # Critical Value (5%)                 0.7860
    # Critical Value (2%)                 0.9170
    # Critical Value (1%)                 1.0900
    # rejection of original hypothesis         1
    print(ks_test(df['close']))
    # KS test statistic                  38.5932
    # p-value                             0.0000
    # p-value significant                 1.0000
    # statistic larger Than               0.0100
    # Critical Value (15%)                0.5750
    # Critical Value (10%)                0.6550
    # Critical Value (5%)                 0.7860
    # Critical Value (2%)                 0.9170
    # Critical Value (1%)                 1.0900
    # rejection of original hypothesis         1
    print(ks_test(df['close'].to_returns().dropna()))
    # KS test statistic                  0.3637
    # p-value                            0.1952
    # p-value significant                0.0000
    # statistic larger Than                None
    # Critical Value (15%)               0.5760
    # Critical Value (10%)               0.6560
    # Critical Value (5%)                0.7870
    # Critical Value (2%)                0.9180
    # Critical Value (1%)                1.0920
    # rejection of original hypothesis        0
    print(ks_test(np.random.normal(0, 1, 10000)))


if __name__ == "__main__":
    _test_adf_test()
    _test_coint_test()
    _test_corr()
    _test_ks_test()
    _test_adf_coint_test()
