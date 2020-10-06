#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/10/6 11:20
@File    : factor_analysis.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo
from numpy.linalg import LinAlgError

import ibats_common.config  # NOQA

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def factor_analysis(factor_df, max_feature_count=None, plot=True):
    """
    因子分析，提取N个特征，查看是否有效
    :param factor_df:
    :param max_feature_count:
    :param plot:
    :return:
    """
    ana_dic = {}
    max_feature_count = np.min([factor_df.shape[1] // 3, 50] if max_feature_count is None else max_feature_count)
    for n_features in range(2, max_feature_count):
        logger.info(f"{n_features} 个因子时:")
        fa = FactorAnalyzer(n_factors=n_features, rotation=None)
        exception = None
        for _ in range(8, 0, -1):
            df = factor_df if _ == 0 else factor_df.sample(factor_df.shape[0] // (_ + 1) * _)
            try:
                fa.fit(df)
                break
            except LinAlgError as exp:
                exception = exp
                logger.exception(
                    "当前矩阵 %s 存在可逆矩阵，尝试进行 %d/(%d+1) 重新采样", df.shape, _, _)
                logger.warning(exception is None)
        else:
            logger.warning(exception is None)
            raise exception from exception

        communalities = fa.get_communalities()
        logger.info(f"\t共因子方差比（communality）({communalities.shape})")  # 公因子方差
        # logger.debug('\n%s', communalities)
        loadings = fa.loadings_
        logger.info(f"\t成分矩阵，即：因子载荷（loading）({loadings.shape})")  # 成分矩阵
        # logger.debug('\n%s', loadings)  # 成分矩阵
        var = fa.get_factor_variance()  # 给出贡献率
        # 1. Sum of squared loadings (variance)
        # 2. Proportional variance
        # 3. Cumulative variance
        logger.info(f"\tCumulative variance {var[2]}")
        kmo_per_variable, kmo_total = calculate_kmo(fa.transform(factor_df))
        if kmo_total < 0.6:
            logger.info(f'\t× -> kmo_total={kmo_total:.5f} 变量间的相关性弱，不适合作因子分析')
        else:
            logger.info(f'\t√ -> kmo_total={kmo_total:.5f} 变量间的相关性强，变量越适合作因子分析')
        ana_dic[n_features] = {
            "FactorAnalyzer": fa,
            # "communalities": communalities,
            # "loadings": loadings,
            # "Sum of squared loadings": var[0],
            # "Proportional variance": var[1],
            "Cumulative variance": var[2][-1],
            "KOM_Test_total": kmo_total,
        }
        if var[2][-1] > 0.95 and kmo_total > 0.6:
            break

    ana_data = pd.DataFrame({k: v for k, v in ana_dic.items() if k != 'FactorAnalyzer'}).T
    if plot:
        ana_data.plot(subplots=True, figsize=(9, 6))
        plt.show()

    return ana_dic


def _test_factor_analysis():
    from ibats_common.example.data import load_data
    from ibats_common.backend.factor import get_factor
    df = load_data(
        "RB.csv", index_col='trade_date', parse_index_to_datetime=True
    ).drop(['instrument_type'], axis=1)
    factor_df = get_factor(df, price_factor_kwargs={'with_diff_n': False}).dropna()
    ana_dic = factor_analysis(factor_df)


if __name__ == "__main__":
    _test_factor_analysis()
