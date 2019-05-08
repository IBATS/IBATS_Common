#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-8 下午2:53
@File    : wave.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
import numpy as np
from ibats_common.analysis.plot import hist_norm
from ibats_common.example.data import load_data
import logging
from scipy.stats import anderson
import scipy.stats as stats


logger = logging.getLogger(__name__)


def wave_hist():
    df = load_data('RB.csv')
    pct_change_s = df['close'].pct_change().dropna()
    logger.info('pct_change description:\n%s', pct_change_s.describe())
    logger.info('pct_change quantile:\n%s', pct_change_s.quantile([_ / 20 for _ in range(20)]))
    data = pct_change_s.to_numpy()
    hist_norm(data, num_bins=50)
    result = stats.normaltest(data)
    logger.info('stats.normaltest: %s', result)
    result = stats.anderson(data)
    logger.info('stats.normaltest: %s', result)


if __name__ == "__main__":
    pass
