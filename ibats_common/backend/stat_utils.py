"""
@author  : MG
@Time    : 2020/9/11 13:49
@File    : stat_utils.py
@contact : mmmaaaggg@163.com
@desc    : 用于提供各种统计学工具
"""
import typing
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

LABEL_T_SMALLER_THAN = 't-value Smaller Than'
LABEL_P_SIGNIFICANT = 'p-value significant'
LABEL_REJECTION_OF_ORIGINAL_HYPOTHESIS = 'rejection of original hypothesis'


def adf_test(data: typing.Union[pd.DataFrame, pd.Series, np.ndarray]) -> typing.Union[pd.Series, pd.DataFrame]:
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
    adf_test(df)
    adf_test((df.pct_change()/df).dropna())


if __name__ == "__main__":
    pass
