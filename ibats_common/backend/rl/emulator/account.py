#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/6/28 17:08
@File    : account.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import numpy as np
import pandas as pd
from ibats_common.backend.rl.emulator.market import QuotesMarket


class Account(object):
    def __init__(self, md_df, data_factors):
        self.A = QuotesMarket(md_df, data_factors)
        self.buffer_reward = []
        self.buffer_value = []
        self.buffer_action = []
        self.buffer_cash = []

    def reset(self):
        self.buffer_reward = []
        self.buffer_value = []
        self.buffer_action = []
        self.buffer_cash = []
        return np.expand_dims(self.A.reset(), 0)

    def latest_state(self):
        return np.expand_dims(self.A.latest_state(), 0)

    def step(self, action):
        next_state, reward, done = self.A.step(action)

        self.buffer_action.append(action)
        self.buffer_reward.append(reward)
        self.buffer_value.append(self.A.total_value)
        self.buffer_cash.append(self.A.cash)
        return np.expand_dims(next_state, 0), reward, done

    def plot_data(self):
        df = pd.DataFrame([self.buffer_value, self.buffer_reward, self.buffer_cash, self.buffer_action]).T
        length = df.shape[0]
        df.index = self.A.data_close.index[:length]
        df.columns = ["value", "reward", "cash", "action"]
        return df


def _test_account():
    # 建立相关数据
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    shape = [data_arr_batch.shape[0], 5, int(n_step/5), data_arr_batch.shape[2]]
    data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    # 建立 Account
    env = Account(md_df, data_factors)
    next_observation = env.reset()
    print('next_observation.shape:', next_observation.shape)
    assert next_observation.shape == (1, 12, 78, 5)
    next_state, reward, done = env.step(1)
    assert next_observation.shape == (1, 12, 78, 5)
    assert not done


if __name__ == "__main__":
    _test_account()
