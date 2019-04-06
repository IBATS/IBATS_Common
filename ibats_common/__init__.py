#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/6/26 10:27
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_utils.mess import get_folder_path
import os

local_model_folder_path = get_folder_path('ibats_common', create_if_not_found=False)
local_root_path = os.path.abspath(
    os.path.join(local_model_folder_path, os.path.pardir))
