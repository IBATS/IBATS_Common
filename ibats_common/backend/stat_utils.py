"""
@author  : MG
@Time    : 2020/9/11 13:49
@File    : stat_utils.py
@contact : mmmaaaggg@163.com
@desc    : 用于提供各种统计学工具
"""
from typing import Union
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

LABEL_T_SMALLER_THAN = 't-statistic Smaller Than'
LABEL_P_SIGNIFICANT = 'p-value significant'
LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS = 'rejection of original hypothesis'


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


if __name__ == "__main__":
    _test_adf_test()
    _test_coint_test()
