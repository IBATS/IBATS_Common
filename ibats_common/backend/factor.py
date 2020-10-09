#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/5/20 21:09
@File    : factor.py
@contact : mmmaaaggg@163.com
@desc    : 用来对时间序列数据建立因子
"""
import datetime
import logging
import bisect
import ffn  # NOQA
import numpy as np
import pandas as pd
import talib
from ibats_utils.mess import date_2_str

logger = logging.getLogger(__name__)
DEFAULT_OHLCV_COL_NAME_LIST = ["open", "high", "low", "close", "amount", "volume"]


def add_factor_of_trade_date(df: pd.DataFrame, trade_date_series):
    """
    自然日，交易日相关因子
    :param df:
    :param trade_date_series:
    :return:
    """
    # 自然日相关因子
    index_s = pd.Series(df.index)
    df['dayofweek'] = index_s.apply(lambda x: x.dayofweek).to_numpy()
    df['day'] = index_s.apply(lambda x: x.day).to_numpy()
    df['month'] = index_s.apply(lambda x: x.month).to_numpy()
    df['daysleftofmonth'] = index_s.apply(lambda x: x.days_in_month - x.day).to_numpy()

    # 本月第几个交易日, 本月还剩几个交易日
    groups = trade_date_series.groupby(trade_date_series.apply(lambda x: datetime.datetime(x.year, x.month, 1)))

    def get_td_of_month(dt):
        first_day_of_month = datetime.datetime(dt.year, dt.month, 1)
        month_s = groups.get_group(first_day_of_month)
        td_of_month = (month_s <= dt).sum()
        td_left_of_month = (month_s > dt).sum()
        return month_s.shape[0], td_of_month, td_left_of_month

    result_arr = index_s.apply(get_td_of_month).to_numpy()
    df['td_of_month'] = [_[0] for _ in result_arr]
    df['td_pass_of_month'] = [_[1] for _ in result_arr]
    df['td_left_of_month'] = [_[2] for _ in result_arr]

    # 本周第几个交易日, 本周还剩几个交易日
    groups = trade_date_series.groupby(trade_date_series.apply(lambda x: x.year * 100 + x.weekofyear))

    def get_td_of_week(dt):
        name = dt.year * 100 + dt.weekofyear
        week_s = groups.get_group(name)
        td_pass_of_week = (week_s <= dt).sum()
        td_left_of_week = (week_s > dt).sum()
        return week_s.shape[0], td_pass_of_week, td_left_of_week

    result_arr = index_s.apply(get_td_of_week).to_numpy()
    df['td_of_week'] = [_[0] for _ in result_arr]
    df['td_pass_of_week'] = [_[1] for _ in result_arr]
    df['td_left_of_week'] = [_[2] for _ in result_arr]

    # 距离下一次放假交易日数（超过2天以上的休息日）
    # 计算距离下一个交易日的日期
    days_2_next_trade_date_s = (trade_date_series.shift(-1) - trade_date_series).fillna(pd.Timedelta(days=0))
    days_2_next_trade_date_s.index = pd.DatetimeIndex(trade_date_series)
    # 倒序循环，计算距离下一次放假的日期
    days_count, trade_date_count, result_dic, first_date = np.nan, np.nan, {}, index_s[0]
    for trade_date, delta in days_2_next_trade_date_s.sort_index(ascending=False).items():
        if trade_date < first_date:
            break
        days = delta.days
        if days > 3:
            trade_date_count, days_count = 0, 0
        elif pd.isna(days_count) >= 0 and days >= 1:
            trade_date_count += 1
            days_count += days
        else:
            trade_date_count, days_count = np.nan, np.nan

        result_dic[trade_date] = [days_count, trade_date_count]

    vacation_df = pd.DataFrame(result_dic).T.sort_index().rename(columns={0: 'days_2_vacation', 1: 'td_2_vacation'})
    df = df.join(vacation_df, how='left')

    return df


def add_factor_of_delivery_date(df, delivery_date_series):
    index_s = pd.Series(df.index)

    # 当期合约距离交割日天数
    result_arr = []
    for _, trade_date in index_s.items():
        next_2_date = delivery_date_series[delivery_date_series > trade_date].head(2)
        if next_2_date.shape[0] < 2:
            day_2_first_del_date, day_2_second_del_date = np.nan, np.nan
        else:
            first_del_date, secend_del_date = next_2_date[0], next_2_date[1]
            day_2_first_del_date = (first_del_date - trade_date).days
            day_2_second_del_date = (secend_del_date - trade_date).days

        result_arr.append([day_2_first_del_date, day_2_second_del_date])

    df['days_2_first_del_date'] = [_[0] for _ in result_arr]
    df['days_2_second_del_date'] = [_[1] for _ in result_arr]
    return df


def add_factor_of_price(df: pd.DataFrame, ohlcav_col_name_list=DEFAULT_OHLCV_COL_NAME_LIST,
                        drop=False, log_av=True, add_pct_change_columns=True, with_diff_n=True):
    """
    计算数据的因子
    :param df:
    :param ohlcav_col_name_list: 各因子列的标签名称，其中 amount 列，如果没有可以为 None
    :param drop:
    :param log_av: 对 amount volume 进行log
    :param add_pct_change_columns: 对非平稳序列增加相应的 pct_change 列
    :param with_diff_n: 增加N阶 diff，增加N阶Diff，
        可能导致 factor_analysis 抛出 LinAlgError，
        因此在进行 factor analysis 时，可以关掉次因子
    :return:
    """
    pct_change_columns = [_ for _ in ohlcav_col_name_list if _ is not None]
    open_key = ohlcav_col_name_list[0]
    high_key = ohlcav_col_name_list[1]
    low_key = ohlcav_col_name_list[2]
    close_key = ohlcav_col_name_list[3]
    amount_key = ohlcav_col_name_list[4]
    volume_key = ohlcav_col_name_list[5]
    open_s = df[open_key]
    high_s = df[high_key]
    low_s = df[low_key]
    close_s = df[close_key]
    amount_s = df[amount_key] if amount_key is not None else None
    volume_s = df[volume_key]
    # 平均成交价格
    if amount_s is None:
        deal_price_s = (open_s * 2 + high_s + low_s + close_s * 2) / 6
    else:
        deal_price_s = amount_s / volume_s
        deal_price_s[volume_s.isna()] = ((open_s * 2 + high_s + low_s + close_s * 2) / 6)[volume_s.isna()]

    df[f'deal_price'] = deal_price_s
    pct_change_columns.append('deal_price')
    # N 阶价差
    if with_diff_n:
        for _ in range(2, 4):
            # 因为 close diff(1) 已经在前面 ohlcav_col_name_list 中完成了对所有原始数据列的一阶差分，因此无需重复
            df[f'diff{_}'] = close_s.diff(_)

    # 均线因子
    df[f'rr'] = close_s.to_returns()
    for n in [5, 10, 15, 20, 30, 60, 120]:
        df[f'MA{n}'] = ma_n = close_s.rolling(n).mean()
        pct_change_columns.append(f'MA{n}')
        df[f'c-MA{n}'] = (close_s - ma_n) / close_s
        for m in [5, 10, 15, 20, 30, 60]:
            # 生成均线差的因子 命名规则 MAm - MAn
            if m >= n:
                continue
            df[f'MA{m}-MA{n}'] = (df[f'MA{m}'] - ma_n) / close_s

    # 波动率因子
    expanding = close_s.expanding(5)
    df[f'volatility_all'] = expanding.std() / expanding.mean()
    for n in [10, 20, 30, 60]:
        df[f'volatility{n}'] = close_s.rolling(n).std() / close_s.rolling(n).mean()

    # 收益率方差
    rr = close_s.to_returns()
    for n in [5, 10, 20, 30, 60]:
        df[f'rr_std{n}'] = rr.rolling(n).std()

    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    df['AD'] = talib.AD(high_s, low_s, close_s, volume_s)
    pct_change_columns.append(f'AD')

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    df['ADOSC'] = talib.ADOSC(high_s, low_s, close_s, volume_s, fastperiod=3, slowperiod=10)
    pct_change_columns.append(f'ADOSC')

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    df['ADX'] = talib.ADX(high_s, low_s, close_s, timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    df['ADXR'] = talib.ADXR(high_s, low_s, close_s, timeperiod=14)

    # 绝对价格振荡指数
    df['APO'] = talib.APO(close_s, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，
    # 帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    df['AROONDown'], df['AROONUp'] = talib.AROON(high_s, low_s, timeperiod=14)
    df['AROONOSC'] = talib.AROONOSC(high_s, low_s, timeperiod=14)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    for n in [6, 14, 21]:
        df[f'ATR{n}'] = talib.ATR(high_s, low_s, close_s, timeperiod=n)

    # 布林带
    df['Boll_Up'], df['Boll_Mid'], df['Boll_Down'] = \
        talib.BBANDS(close_s, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    pct_change_columns.extend(['Boll_Up', 'Boll_Mid', 'Boll_Down'])

    # 均势指标
    df['BOP'] = talib.BOP(open_s, high_s, low_s, close_s)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    for n in [5, 10, 20, 88]:
        df[f'CCI{n}'] = talib.CCI(high_s, low_s, close_s, timeperiod=5)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如
    # 相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    df['CMO_Close'] = talib.CMO(close_s, timeperiod=14)
    df['CMO_Open'] = talib.CMO(open_s, timeperiod=14)

    # DEMA双指数移动平均线
    for n in [6, 12, 26]:
        df[f'DEMA{n}'] = talib.DEMA(close_s, timeperiod=n)
        pct_change_columns.append(f'DEMA{n}')

    # DX 动向指数
    df['DX'] = talib.DX(high_s, low_s, close_s, timeperiod=14)

    # EMA 指数移动平均线
    for n in [6, 12, 26, 60]:
        df[f'EMA{n}'] = talib.EMA(close_s, timeperiod=n)
        pct_change_columns.append(f'EMA{n}')

    # KAMA 适应性移动平均线
    for n in [5, 10, 20, 30, 60]:
        df[f'KAMA{n}'] = talib.KAMA(close_s, timeperiod=n)
        pct_change_columns.append(f'KAMA{n}')

    # MACD
    df['MACD_DIF'], df['MACD_DEA'], df['MACD_bar'] = \
        talib.MACD(close_s, fastperiod=12, slowperiod=24, signalperiod=9)

    # 中位数价格 不知道是什么意思
    df['MEDPRICE'] = talib.MEDPRICE(high_s, low_s)
    pct_change_columns.append('MEDPRICE')

    # 负向指标 负向运动
    df['MiNUS_DI'] = talib.MINUS_DI(high_s, low_s, close_s, timeperiod=14)
    df['MiNUS_DM'] = talib.MINUS_DM(high_s, low_s, timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    df['MOM'] = talib.MOM(close_s, timeperiod=10)

    # 归一化平均值范围
    df['NATR'] = talib.NATR(high_s, low_s, close_s, timeperiod=14)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    df['OBV'] = talib.OBV(close_s, volume_s)
    pct_change_columns.append('OBV')

    # PLUS_DI 更向指示器
    df['PLUS_DI'] = talib.PLUS_DI(high_s, low_s, close_s, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high_s, low_s, timeperiod=14)

    # PPO 价格振荡百分比
    df['PPO'] = talib.PPO(close_s, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    for n in [6, 20]:
        df[f'ROC{n}'] = talib.ROC(close_s, timeperiod=n)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    for n in [6, 20]:
        df[f'VROC{n}'] = talib.ROC(volume_s, timeperiod=n)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    for n in [6, 20]:
        df[f'ROCP{n}'] = talib.ROCP(close_s, timeperiod=n)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    for n in [6, 20]:
        df[f'VROCP{n}'] = talib.ROCP(volume_s, timeperiod=n)

    # RSI
    for n in [7, 14, 21, 28]:
        df[f'RSI{n}'] = talib.RSI(close_s, timeperiod=n)

    # SAR 抛物线转向
    df['SAR'] = talib.SAR(high_s, low_s, acceleration=0.02, maximum=0.2)
    pct_change_columns.append('SAR')

    # TEMA
    for n in [6, 12, 26]:
        df[f'TEMA{n}'] = talib.TEMA(close_s, timeperiod=n)
        pct_change_columns.append(f'TEMA{n}')

    # TRANGE 真实范围
    df['TRANGE'] = talib.TRANGE(high_s, low_s, close_s)

    # TYPPRICE 典型价格
    df['TYPPRICE'] = talib.TYPPRICE(high_s, low_s, close_s)
    pct_change_columns.append('TYPPRICE')

    # TSF 时间序列预测
    for n in [7, 14, 21, 28]:
        df[f'TSF{n}'] = talib.TSF(close_s, timeperiod=n)
        pct_change_columns.append(f'TSF{n}')

    # ULTOSC 极限振子
    df['ULTOSC'] = talib.ULTOSC(high_s, low_s, close_s, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    df['WILLR'] = talib.WILLR(high_s, low_s, close_s, timeperiod=14)

    # 价格分位数水平
    data_list, data_count = [], [0]

    def get_index_pct(x):
        """获取当前价格在历史价格数组中的位置的百分比"""
        bisect.insort(data_list, x)
        data_count[0] += 1
        idx = bisect.bisect_left(data_list, x)
        return idx / data_count[0]

    df['index_pct'] = close_s.apply(get_index_pct)
    pct_change_columns.append('index_pct')

    # 对 volume amount 取 log
    if log_av:
        df[volume_key] = np.log(volume_s.fillna(0) + 1)
        if amount_key is not None:
            df[amount_key] = np.log(amount_s.fillna(0) + 1)

    # 对非平稳的序列因子进行 pct_change 或 diff(1)一阶差分 处理，期待可以形成新的平稳的序列
    if add_pct_change_columns:
        for name in pct_change_columns:
            _s = df[name]
            if _s.std() < 100:
                values = _s.diff(1)
                df[f'{name}_diff1'] = values
            else:
                values = df[name].pct_change()
                is_inf_values = np.isneginf(values)
                values[is_inf_values] = np.min(values[~is_inf_values])
                is_inf_values = np.isposinf(values)
                values[is_inf_values] = np.max(values[~is_inf_values])
                df[f'{name}_pct_chg'] = values

    if drop:
        df.dropna(inplace=True)

    return df


def get_factor(df: pd.DataFrame, trade_date_series=None, delivery_date_series=None,
               dropna=True, ohlcav_col_name_list=["open", "high", "low", "close", "amount", "volume"],
               do_multiple_factors=False, price_factor_kwargs=None) -> (pd.DataFrame, dict):
    """
    在当期时间序列数据基础上增加相关因子
    目前已经支持的因子包括量价因子、时间序列因子、交割日期因子
    :param df: 时间序列数据索引为日期
    :param trade_date_series: 交易日序列
    :param delivery_date_series:交割日序列
    :param dropna: 是否 dropna
    :param ohlcav_col_name_list: 量价因子相关列名称，默认["open", "high", "low", "close", "amount", "volume"]
    :param do_multiple_factors: 对数据进行倍增处理，将指定列乘以因子，如果 ！= None 则，返回dict{adj_factor: DataFrame}
    :param price_factor_kwargs: 计算价格因子时的可选参数
    :return:
    """
    price_factor_kwargs = {} if price_factor_kwargs is None else price_factor_kwargs
    ret_df = df.copy()

    # 交易日相关因子
    if trade_date_series is not None:
        ret_df = add_factor_of_trade_date(ret_df, trade_date_series)

    # 交割日相关因子
    if delivery_date_series is not None:
        ret_df = add_factor_of_delivery_date(ret_df, delivery_date_series)

    # 增加量价因子
    train_df_dic = {}
    if ohlcav_col_name_list is not None:
        ret_df_tmp = ret_df.copy()
        ret_df = add_factor_of_price(ret_df, ohlcav_col_name_list, **price_factor_kwargs)
        if do_multiple_factors:
            if dropna:
                ret_df.dropna(inplace=True)
            train_df_dic[1] = ret_df
            for adj_factor in [0.5, 0.75, 1.25, 1.5, 1.75, 2]:
                train_df_tmp = ret_df_tmp.copy()
                # 将 O,H,L,C,A 前五项进行因子扩充
                train_df_tmp.loc[:, ohlcav_col_name_list[:5]] *= adj_factor
                factor_df = add_factor_of_price(
                    train_df_tmp, ohlcav_col_name_list=ohlcav_col_name_list, **price_factor_kwargs)
                if dropna:
                    factor_df.dropna(inplace=True)
                train_df_dic[adj_factor] = factor_df

    if do_multiple_factors:
        return train_df_dic
    else:
        if dropna:
            ret_df.dropna(inplace=True)

        return ret_df


def _test_get_factor(do_multiple_factors=False):
    from ibats_common.example.data import load_data, get_trade_date_series, get_delivery_date_series
    instrument_type = 'RU'
    file_name = f"{instrument_type}.csv"
    indexed_df = load_data(file_name).set_index('trade_date').drop('instrument_type', axis=1)
    indexed_df.index = pd.DatetimeIndex(indexed_df.index)
    # ohlcav_col_name_list is None
    factor_df = get_factor(indexed_df,
                           trade_date_series=get_trade_date_series(),
                           delivery_date_series=get_delivery_date_series(instrument_type))
    logger.info("data_multiplication_column_indexes=None\n%s\t%s", factor_df.shape, list(factor_df.columns))

    # ohlcav_col_name_list is not None
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    train_df_dic = get_factor(
        indexed_df, ohlcav_col_name_list=ohlcav_col_name_list,
        trade_date_series=get_trade_date_series(),
        delivery_date_series=get_delivery_date_series(instrument_type),
        do_multiple_factors=do_multiple_factors
    )
    logger.info("ohlcav_col_name_list=%s", ohlcav_col_name_list)
    if do_multiple_factors:
        for adj_factor, train_df in train_df_dic.items():
            logger.info("adj_factor=%f, train_df: first close=%f\n%s\t%s",
                        adj_factor, train_df['close'].iloc[0], train_df.shape, list(train_df.columns))
    else:
        train_df = train_df_dic
        logger.info("train_df: first close=%f\n%s\t%s",
                    train_df['close'].iloc[0], train_df.shape, list(train_df.columns))
        idxs = np.where(np.isnan(train_df))
        if len(idxs[0]) > 0:
            logger.error("has nan value at %s", idxs)

        idxs = np.where(np.isinf(train_df))
        if len(idxs[0]) > 0:
            logger.error("has inf value at %s", idxs)

        pass


def transfer_2_batch(df: pd.DataFrame, n_step, labels=None, date_from=None, date_to=None):
    """
    [num, factor_count] -> [num - n_step + 1, n_step, factor_count]
    将 df 转化成 n_step 长度的一段一段的数据
    labels 为与 df对应的数据，处理方式与index相同，如果labels不为空，则返回数据最后增加以下 new_ys
    :param df:
    :param n_step:
    :param labels:如果不为 None，则长度必须与 df.shape[0] 一致
    :param date_from:
    :param date_to:
    :return:
    """
    df_len = df.shape[0]
    if labels is not None and df_len != len(labels):
        raise ValueError("ys 长度 %d 必须与 df 长度 %d 保持一致", len(labels), df_len)
    # TODO: date_from, date_to 的逻辑可以进一步优化，延期为了省时间先保持这样
    # 根据 date_from 对factor进行截取
    if date_from is not None:
        date_from = pd.to_datetime(date_from)
        is_fit = df.index >= date_from
        if np.any(is_fit):
            start_idx = np.argmax(is_fit) - n_step

            if start_idx < 0:
                start_idx = 0
                logger.warning("%s 为起始日期的数据，前向历史数据不足 %d 条，因此，起始日期向后推移至 %s",
                               date_2_str(date_from), n_step, date_2_str(df.index[60]))

            df = df.iloc[start_idx:]
            df_len = df.shape[0]
            if labels is not None:
                labels = labels[start_idx:]

        else:
            logger.warning("没有 %s 之后的数据，当前数据最晚日期为 %s",
                           date_2_str(date_from), date_2_str(max(df.index)))
            if labels is not None:
                return None, None, None, None
            else:
                return None, None, None

    # 根据 date_from 对factor进行截取
    if date_to is not None:
        date_to = pd.to_datetime(date_to)
        is_fit = df.index <= date_to
        if np.any(is_fit):
            to_idx = np.argmin(is_fit)
            df = df.iloc[:to_idx]
            df_len = df.shape[0]
            if labels is not None:
                labels = labels[:to_idx]

        else:
            logger.warning("没有 %s 之前的数据，当前数据最晚日期为 %s",
                           date_2_str(date_to), date_2_str(min(df.index)))
            if labels is not None:
                return None, None, None, None
            else:
                return None, None, None

    new_shape = [df_len - n_step + 1, n_step]
    new_shape.extend(df.shape[1:])
    df_index, df_columns = df.index[(n_step - 1):], df.columns
    data_arr_batch, factor_arr = np.zeros(new_shape), df.to_numpy(dtype=np.float32)

    for idx_from, idx_to in enumerate(range(n_step, factor_arr.shape[0] + 1)):
        data_arr_batch[idx_from] = factor_arr[idx_from: idx_to]

    if labels is not None:
        new_ys = labels[(n_step - 1):]
        return df_index, df_columns, data_arr_batch, new_ys
    else:
        return df_index, df_columns, data_arr_batch


def _test_get_batch_factor():
    data_len = 8
    date_arr = pd.date_range(pd.to_datetime('2018-01-01'),
                             pd.to_datetime('2018-01-01') + pd.Timedelta(days=data_len * 2 - 1),
                             freq=pd.Timedelta(days=2))
    date_index = pd.DatetimeIndex(date_arr)
    df = pd.DataFrame(
        {'a': list(range(data_len)),
         'b': list(range(data_len * 2, data_len * 3)),
         'c': list(range(data_len * 10, data_len * 11))},
        index=date_index,
    )
    labels = list(range(data_len))
    print("df\n", df)
    n_step = 5
    df_index, df_columns, data_arr_batch, new_labels = transfer_2_batch(df, n_step, labels)
    print("new df_index", df_index)
    print("new factor_columns", df_columns)
    print('new reshaped data_arr_batch')
    print(data_arr_batch)
    print("df.shape: ", df.shape)
    print("new_shape:", data_arr_batch.shape)
    print("new_labels:", new_labels)

    date_from = '2018-01-13'
    df_index, df_columns, data_arr_batch, new_labels = transfer_2_batch(df, n_step, labels, date_from=date_from)
    print('date_from=', date_from)
    print("new df_index", df_index)
    print("new factor_columns", df_columns)
    print('new reshaped data_arr_batch')
    print(data_arr_batch)
    print("df.shape: ", df.shape)
    print("new_shape:", data_arr_batch.shape)
    print("new_labels:", new_labels)


def get_batch_factor(md_df: pd.DataFrame, n_step, labels=None, **factor_kwargs):
    """
    get_factor(...) -> transfer_2_batch(...)
    :param md_df:
    :param n_step:
    :param labels:
    :param factor_kwargs:
    :return:
    """
    factor_df = get_factor(md_df, **factor_kwargs)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factor_df, n_step=n_step, labels=labels)
    return df_index, df_columns, data_arr_batch


if __name__ == '__main__':
    from ibats_common.config import *

    logger = logging.getLogger()
    _test_get_factor()
    # _test_get_batch_factor()
