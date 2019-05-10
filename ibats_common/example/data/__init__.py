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
from ibats_common import module_root_path


def load_data(file_name, encoding=None)-> pd.DataFrame:
    file_path = os.path.join(module_root_path, 'example', 'data', file_name)
    df = pd.read_csv(file_path, encoding=encoding)
    return df


if __name__ == "__main__":
    pass
