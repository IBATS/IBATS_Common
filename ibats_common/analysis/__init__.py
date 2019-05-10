#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:25
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
from functools import lru_cache

pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:,.4f}'.format)


@lru_cache(maxsize=1)
def get_report_folder_path() -> str:
    import os
    from ibats_common import module_root_path
    _report_file_path = os.path.join(module_root_path, 'analysis', 'report')
    if not os.path.exists(_report_file_path):
        os.makedirs(_report_file_path)

    return _report_file_path


@lru_cache(maxsize=1)
def get_cache_folder_path() -> str:
    import os
    from ibats_common import module_root_path
    _report_file_path = os.path.join(module_root_path, 'analysis', 'report', "_cache_")
    if not os.path.exists(_report_file_path):
        os.makedirs(_report_file_path)

    return _report_file_path

