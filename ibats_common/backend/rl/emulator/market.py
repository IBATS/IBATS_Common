#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/6/28 17:09
@File    : market.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import numpy as np
import pandas as pd


class Action:
    close = 0
    long = 1
    short = 2
    keep = 3


class QuotesMarket(object):
    def __init__(self, md_df: pd.DataFrame, data_factors, init_cash=2e5, fee_rate=3e-3,
                 state_with_flag=False, reward_with_fee0=False, position_count=10):
        self.data_close = md_df['close']
        self.data_open = md_df['open']
        self.data_observation = data_factors
        self.action_space = ['close', 'long', 'short', 'keep']
        self.fee_rate = fee_rate  # 千三手续费
        self.fee_curr_step = 0
        self.fee_tot = 0
        self.max_step_count = self.data_observation.shape[0] - 1
        self.init_cash = init_cash
        # reset use
        self.step_counter = 0
        self.cash = self.init_cash
        self.holding_value = 0      # holding_value 多仓为整，空仓为负
        self.total_value = self.cash + self.holding_value
        self._open_curr = None
        self.total_value_fee0 = self.cash + self.holding_value
        self.flags = 0
        self.state_with_flag = state_with_flag
        self.reward_with_fee0 = reward_with_fee0
        self.action_count = 0
        self.position_count = position_count

    def reset(self):
        self.step_counter = 0
        self.cash = self.init_cash
        self.holding_value = 0
        self.total_value = self.cash + self.holding_value
        self.total_value_fee0 = self.cash + self.holding_value
        self.flags = 0
        self.fee_curr_step = 0
        self.fee_tot = 0
        self.action_count = 0
        if self.state_with_flag:
            return self.data_observation[0], np.array([self.flags])
        else:
            return self.data_observation[0]

    def latest_state(self):
        if self.state_with_flag:
            return self.data_observation[-1], np.array([self.flags])
        else:
            return self.data_observation[-1]

    def get_action_space(self):
        return self.action_space

    def long(self):
        self.flags = 1
        if self._open_curr is None:
            self._open_curr = self.data_open[self.step_counter]
        quotes = self._open_curr * self.position_count
        self.cash -= quotes * (1 + self.fee_rate)
        self.holding_value = quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def short(self):
        self.flags = -1
        if self._open_curr is None:
            self._open_curr = self.data_open[self.step_counter]
        quotes = self._open_curr * self.position_count
        self.cash -= quotes * (1 + self.fee_rate)
        self.holding_value = - quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def keep(self):
        quotes = self.data_open[self.step_counter] * self.position_count
        self.holding_value = quotes * self.flags

    def close_long(self):
        self.flags = 0
        if self._open_curr is None:
            self._open_curr = self.data_open[self.step_counter]
        quotes = self._open_curr * self.position_count
        self.cash += quotes * (1 - self.fee_rate)
        self.holding_value = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def close_short(self):
        self.flags = 0
        if self._open_curr is None:
            self._open_curr = self.data_open[self.step_counter]
        quotes = self._open_curr * self.position_count
        self.cash += quotes * (1 - self.fee_rate)
        self.holding_value = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def step_op(self, action):
        self._open_curr = self.data_open[self.step_counter]
        self.fee_curr_step = 0
        if action == Action.long:
            if self.flags == 0:
                self.long()
            elif self.flags == -1:
                self.close_short()
                self.long()
            else:
                self.keep()

        elif action == Action.close:
            if self.flags == 1:
                self.close_long()
            elif self.flags == -1:
                self.close_short()
            else:
                pass

        elif action == Action.short:
            if self.flags == 0:
                self.short()
            elif self.flags == 1:
                self.close_long()
                self.short()
            else:
                self.keep()
        elif action == Action.keep:
            self.keep()
        else:
            raise ValueError("action should be elements of ['long', 'short', 'close']")

        # 计算价值
        price = self.data_close[self.step_counter]
        holding_value_abs = price * self.position_count
        self.holding_value = holding_value_abs * self.flags
        reward = self.cash + holding_value_abs - self.total_value

        # 计算费用
        self.fee_tot += self.fee_curr_step
        self.total_value_fee0 = self.total_value + self.fee_tot

        # 获得下一状态
        self.step_counter += 1
        self._open_curr = None
        done = False
        if self.total_value < price:
            done = True
        if self.step_counter >= self.max_step_count:
            done = True

        self.total_value = holding_value_abs + self.cash
        next_observation = self.data_observation[self.step_counter]

        ret_state = (next_observation, np.array([self.flags])) if self.state_with_flag else next_observation
        divisor = price * self.position_count
        ret_reward = (reward / divisor, (reward + self.fee_curr_step) / divisor) if self.reward_with_fee0 else (
                reward / divisor)

        return ret_state, ret_reward, done

    def step(self, action):
        if 0 <= action <= 3:
            return self.step_op(action)
        else:
            raise ValueError("action should be one of [0,1,2]")


def _test_quote_market():
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    # 建立 QuotesMarket
    qm = QuotesMarket(md_df=md_df[['close', 'open']], data_factors=data_arr_batch, state_with_flag=True)
    next_observation = qm.reset()
    assert len(next_observation) == 2
    assert next_observation[0].shape[0] == n_step
    assert next_observation[1] == 0
    next_observation, reward, done = qm.step(1)
    assert len(next_observation) == 2
    assert next_observation[1] == 1
    assert not done
    next_observation, reward, done = qm.step(0)
    assert next_observation[1] == 0
    assert reward != 0
    next_observation, reward, done = qm.step(0)
    assert next_observation[1] == 0
    assert reward == 0
    next_observation, reward, done = qm.step(3)
    assert next_observation[1] == 0
    assert reward == 0
    next_observation, reward, done = qm.step(2)
    assert next_observation[1] == -1
    assert not done
    next_observation, reward, done = qm.step(3)
    assert next_observation[1] == -1
    assert reward != 0
    try:
        qm.step(4)
    except ValueError:
        print('is ok for not supporting action>3')


def _test_market_action():
    """_test_market_action
    测试 action 不同动作下，各中结果是否符合预期
    设计环境（md_df)
    open 价格 10 ～ 15 ~ 10 先增后降，每次涨 1
    close 逢单数步 open + 1 ; 逢双数步 close - 1

    """
    open_price = list(range(10, 15))
    open_price.extend(range(15, 9, -1))
    close_price = [_ + 1 if num % 2 == 0 else _ - 1 for num, _ in enumerate(open_price)]
    md_df = pd.DataFrame({'open': open_price, 'close': close_price},
                         index=pd.date_range('2018-01-01', periods=len(open_price), freq='1D')
                         )[['open', 'close']]
    print(md_df)
    qm = QuotesMarket(md_df=md_df, data_factors=md_df.to_numpy(), state_with_flag=True, fee_rate=1e-3)
    idx_open, idx_close = 0, 1
    next_observation = qm.reset()
    assert len(next_observation) == 2
    assert next_observation[0].shape[0] == 2
    assert next_observation[1] == 0
    (next_observation, flag), reward, done = qm.step(Action.long)
    assert flag == 1
    assert qm.holding_value == 110
    assert qm.fee_tot == 0.1
    assert qm.fee_curr_step == 0.1
    assert 0.0899 < reward < 0.09
    (next_observation, flag), reward, done = qm.step(Action.short)
    assert flag == -1
    assert qm.holding_value == -100
    assert qm.fee_tot == 0.32
    assert qm.fee_curr_step == 0.22
    assert -0.11 < reward < -0.10
    (next_observation, flag), reward, done = qm.step(Action.short)
    assert flag == -1
    assert qm.holding_value == -130
    assert qm.fee_tot == 0.32
    assert qm.fee_curr_step == 0
    assert reward == 3/13
    (next_observation, flag), reward, done = qm.step(Action.long)
    assert flag == 1
    assert qm.holding_value == 120
    assert qm.fee_tot == 0.58
    assert qm.fee_curr_step == 0.26
    assert -1/12 - 0.003 < reward < -1/12


if __name__ == "__main__":
    # _test_quote_market()
    _test_market_action()
