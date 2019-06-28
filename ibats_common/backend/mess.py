#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-4-24 上午7:55
@File    : mess.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
from sqlalchemy.sql import func
from ibats_utils.db import with_db_session, get_db_session, execute_scalar
from ibats_common.backend.orm import StgRunStatusDetail, OrderDetail, TradeDetail, StgRunInfo
from ibats_common.backend import engines


def csv_formatter(file_path_from, file_path_to):
    """
    将 data_integration_celery/tasks/wind/future_reorg/reorg_md_2_db.py 导出的文件重新格式化
    原始文件包括的字段有：
    trade_date,Contract,ContractNext,Close,CloseNext,TermStructure,Volume,VolumeNext,OI,OINext,
    Open,OpenNext,High,HighNext,Low,LowNext,WarehouseWarrant,WarehouseWarrantNext,
    adj_factor_main,adj_factor_secondary,instrument_type
    :param file_path_from:
    :param file_path_to:
    :return:
    """
    df = pd.read_csv(file_path_from)
    col_name_list = ["instrument_type", "trade_date", "Open", "High", "Low", "Close", "Volume",
                     "OI", "WarehouseWarrant", "TermStructure"]
    ret_df = df[col_name_list].rename(columns={_: _.lower() for _ in col_name_list})
    ret_df.to_csv(file_path_to, index=False)


def get_stg_run_info(stg_run_id=None) -> StgRunInfo:
    """
    查询数据库获取最新的 stg_run_id
    :param stg_run_id:
    :return:
    """
    engine_ibats = engines.engine_ibats
    # 获取 收益曲线
    with with_db_session(engine_ibats) as session:
        if stg_run_id is None:
            stg_run_id = session.query(func.max(StgRunInfo.stg_run_id)).scalar()

        info = session.query(StgRunInfo).filter(StgRunInfo.stg_run_id == stg_run_id).first()

    return info


def get_stg_run_id_latest():
    """获取最新的 stg_run_id"""
    engine_ibats = engines.engine_ibats
    with with_db_session(engine_ibats) as session:
        stg_run_id = session.query(func.max(StgRunInfo.stg_run_id)).scalar()

    return stg_run_id


if __name__ == "__main__":
    import os
    folder_path_from = "/home/mg/Downloads/commodity_daily"
    file_name = "RB ConInfoFull_Adj.csv"
    file_path_from = os.path.join(folder_path_from, file_name)
    folder_path_to = "/home/mg/github/IBATS_Common/ibats_common/example/data"
    file_path_to = os.path.join(folder_path_to, f"{file_name.split(' ')[0]}.csv")
    csv_formatter(file_path_from, file_path_to)
