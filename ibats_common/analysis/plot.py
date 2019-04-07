#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/7 16:31
@File    : plot.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_utils.db import with_db_session, get_db_session
from ibats_common.backend import engines
import pandas as pd
from ibats_common.backend.orm import StgRunStatusDetail
import matplotlib.pyplot as plt


def show_cash_and_margin(stg_run_id):
    # stg_run_id=154
    engine_ibats = engines.engine_ibats
    # session = get_db_session(engine_ibats)
    with with_db_session(engine_ibats) as session:
        sql_str = str(
            session.query(
                StgRunStatusDetail.trade_dt.label('trade_dt'),
                StgRunStatusDetail.cash_and_margin.label('cash_and_margin'),
            ).filter(
                StgRunStatusDetail.stg_run_id == stg_run_id
            )
        )

    df = pd.read_sql(sql_str, engine_ibats, params=[stg_run_id], index_col=['trade_dt'])
    df.plot()
    plt.show()


def show_order(stg_run_id):
    pass
