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

VERSION_V1 = 'v1'
VERSION_V2 = 'v2'


class Account(object):
    def __init__(self, md_df, data_factors, expand_dims=True, state_with_flag=False, version=VERSION_V1, **kwargs):
        if version == VERSION_V1:
            from ibats_common.backend.rl.emulator.market import QuotesMarket, ACTION_CLOSE
        elif version == VERSION_V2:
            from ibats_common.backend.rl.emulator.market2 import QuotesMarket, ACTION_CLOSE
        else:
            raise ValueError(f'param version can only be one of {(VERSION_V1, VERSION_V2)}')
        self.A = QuotesMarket(md_df, data_factors, state_with_flag=state_with_flag, **kwargs)
        self.buffer_reward = [0.0]
        self.buffer_value = [self.A.total_value]
        self.buffer_action = [ACTION_CLOSE]
        self.buffer_cash = [self.A.init_cash]
        self.buffer_fee = [0.0]
        self.buffer_value_fee0 = [self.A.total_value]
        self.buffer_fee_tot = [0.0]
        self.buffer_action_count = [0]
        self.expand_dims = expand_dims
        self.actions = self.A.get_action_operations()
        self.action_size = len(self.actions)
        self.state_with_flag = state_with_flag
        self.version = version

    def reset(self):
        if self.expand_dims:
            # return np.expand_dims(self.A.reset(), 0)
            state = self.A.reset()
            if self.state_with_flag:
                init_state = np.expand_dims(state[0], 0), state[1]
            else:
                init_state = np.expand_dims(state, 0)
        else:
            init_state = self.A.reset()

        if self.version == VERSION_V1:
            from ibats_common.backend.rl.emulator.market import ACTION_CLOSE
        elif self.version == VERSION_V2:
            from ibats_common.backend.rl.emulator.market2 import ACTION_CLOSE
        else:
            raise ValueError(f'param version can only be one of {(VERSION_V1, VERSION_V2)}')

        self.buffer_reward = [0.0]
        self.buffer_value = [self.A.total_value]
        self.buffer_action = [ACTION_CLOSE]
        self.buffer_cash = [self.A.init_cash]
        self.buffer_fee = [0.0]
        self.buffer_value_fee0 = [self.A.total_value]
        self.buffer_fee_tot = [0.0]
        self.buffer_action_count = [0]
        return init_state

    def latest_state(self):
        if self.expand_dims:
            # return np.expand_dims(self.A.latest_state(), 0)
            state = self.A.latest_state()
            if self.state_with_flag:
                return np.expand_dims(state[0], 0), state[1]
            else:
                return np.expand_dims(state, 0)
        else:
            return self.A.latest_state()

    def step(self, action):
        next_state, reward, done = self.A.step(action)

        self.buffer_action.append(action)
        self.buffer_reward.append(reward)
        self.buffer_value.append(self.A.total_value)
        self.buffer_cash.append(self.A.cash)
        self.buffer_fee_tot.append(self.A.fee_tot)
        self.buffer_value_fee0.append(self.A.total_value_fee0)
        self.buffer_action_count.append(self.A.action_count)
        if self.expand_dims:
            if self.state_with_flag:
                return (np.expand_dims(next_state[0], 0), next_state[1]), reward, done
            else:
                return np.expand_dims(next_state, 0), reward, done
        else:
            return next_state, reward, done

    def plot_data(self) -> pd.DataFrame:
        return self.generate_reward_df()

    def generate_reward_df(self) -> pd.DataFrame:
        reward_df = pd.DataFrame(
            {"value": self.buffer_value,
             "reward": self.buffer_reward,
             "cash": self.buffer_cash,
             "action": self.buffer_action,
             "open": self.A.data_open.iloc[:len(self.buffer_action)],
             "close": self.A.data_close.iloc[:len(self.buffer_action)],
             "fee_tot": self.buffer_fee_tot,
             "value_fee0": self.buffer_value_fee0,
             "action_count": self.buffer_action_count,
             }, index=self.A.data_close.index[:len(self.buffer_action)])
        cum_rr_s = reward_df['value'] / self.A.init_cash - 1
        reward_df['nav'] = cum_linear_rr_2_cum_exp_rr(cum_rr_s)
        cum_rr_s = reward_df['value_fee0'] / self.A.init_cash - 1
        reward_df['nav_fee0'] = cum_linear_rr_2_cum_exp_rr(cum_rr_s)
        return reward_df


def cum_linear_rr_2_cum_exp_rr(cum_rr_s: pd.Series):
    """将 累计线性增长率 曲线转化为 累计指数增长率 曲线"""
    rr = cum_rr_s.copy()
    rr[1:] -= rr.shift(1)[1:]
    rr = rr + 1
    cum_exp_rr = rr.cumprod()
    return cum_exp_rr


def _test_account():
    """测试 env.reset() 返回状态 是否符合预期"""
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    # 建立 Account
    env = Account(md_df, data_factors)
    next_observation = env.reset()
    print('next_observation.shape:', next_observation.shape)
    assert next_observation.shape == (1, 12, 79, 5)
    next_state, reward, done = env.step(1)
    assert next_observation.shape == (1, 12, 79, 5)
    assert not done


def _test_account2():
    """测试 plot_data 返回数据是否符合预期"""
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    # 建立 Account
    env = Account(md_df, data_factors)
    next_observation = env.reset()
    # 做空
    env.step(2)
    for n in range(int(md_df.shape[0] / 2)):
        env.step(3)
    # 做多
    next_observation, reward, done = env.step(1)
    while not done:
        next_observation, reward, done = env.step(3)

    # 展示结果
    reward_df = env.plot_data()
    value_s = reward_df.iloc[:, 0]
    from ibats_utils.mess import datetime_2_str
    from datetime import datetime
    dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H_%M_%S')
    title = f'test_account_{dt_str}'
    from ibats_common.analysis.plot import plot_twin
    plot_twin(value_s, md_df["close"], name=title)


if __name__ == "__main__":
    _test_account()
    _test_account2()
