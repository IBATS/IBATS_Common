#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-4-30 上午9:36
@File    : __init__.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import os
import pandas as pd
import numpy as np
from ibats_utils.mess import str_2_date

from ibats_common import module_root_path


def load_data(file_name, encoding=None)-> pd.DataFrame:
    file_path = os.path.join(module_root_path, 'example', 'data', file_name)
    df = pd.read_csv(file_path, encoding=encoding)
    return df


def get_trade_date_series():
    df = load_data('trade_date.csv').astype('datetime64[ns]')
    # ret_list = [str_2_date(_) for _ in load_data('trade_date.csv').T.to_numpy()[0]]
    trade_date_s = df.iloc[:, 0]
    return trade_date_s


if __name__ == "__main__":
    pass
