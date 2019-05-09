#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-4-30 上午9:33
@File    : corr.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import pandas as pd
from ibats_common.example.data import load_data
"""
Available options:

compute.[use_bottleneck, use_numexpr]
display.[chop_threshold, colheader_justify, column_space, date_dayfirst, date_yearfirst, encoding, expand_frame_repr, float_format]
display.html.[border, table_schema, use_mathjax]
display.[large_repr]
display.latex.[escape, longtable, multicolumn, multicolumn_format, multirow, repr]
display.[max_categories, max_columns, max_colwidth, max_info_columns, max_info_rows, max_rows, max_seq_items, memory_usage, multi_sparse, notebook_repr_html, pprint_nest_depth, precision, show_dimensions]
display.unicode.[ambiguous_as_wide, east_asian_width]
display.[width]
html.[border]
io.excel.xls.[writer]
io.excel.xlsm.[writer]
io.excel.xlsx.[writer]
io.hdf.[default_format, dropna_table]
io.parquet.[engine]
mode.[chained_assignment, sim_interactive, use_inf_as_na, use_inf_as_null]
plotting.matplotlib.[register_converters]
"""


def corr(df: pd.DataFrame, add_additional_cols=True):
    if add_additional_cols:
        data_df = df.copy()
        data_df['open_t1'] = data_df['open'].shift(-1)
        data_df['high_t1'] = data_df['high'].shift(-1)
        data_df['low_t1'] = data_df['low'].shift(-1)
        data_df['close_t1'] = data_df['close'].shift(-1)
        data_df['ma5'] = data_df['close'].rolling(window=5).mean()
        data_df['ma10'] = data_df['close'].rolling(window=10).mean()
        data_df['ma20'] = data_df['close'].rolling(window=20).mean()
        data_df['pct_change_vol'] = data_df['volume'].pct_change()
        data_df['pct_change'] = data_df['close'].pct_change()
        data_df['pct_change_T1'] = data_df['pct_change'].shift(-1)
    else:
        data_df = df

    corr_df = data_df.dropna().corr()
    return corr_df


if __name__ == "__main__":
    df = load_data('RB.csv')
    corr(df)
