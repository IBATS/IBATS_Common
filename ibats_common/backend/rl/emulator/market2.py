#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/6/28 17:09
@File    : market.py
@contact : mmmaaaggg@163.com
@desc    :
2020-01-29
在 market 基础上进行修改
调整 action 数值，为了更加适应 one_hot 模式，调整 long=0, short=1 close=2, keep=3
"""
import numpy as np
import pandas as pd

ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP = 0, 1, 2, 3
ACTIONS = [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP]
ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP = 'long', 'short', 'close', 'keep'
ACTION_OPS = [ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP]
ACTION_OP_DIC = {_: ACTION_OPS[_] for _ in ACTIONS}
OP_ACTION_DIC = {ACTION_OPS[_]: _ for _ in ACTIONS}
# 内部访问的flag
_FLAG_LONG, _FLAG_SHORT, _FLAG_EMPTY = 1, -1, 0
# one-hot 模式的flag
FLAG_LONG, FLAG_SHORT, FLAG_EMPTY = _FLAG_LONG + 1, _FLAG_SHORT + 1, _FLAG_EMPTY + 1
FLAGS = [FLAG_LONG, FLAG_SHORT, FLAG_EMPTY]


class QuotesMarket(object):
    def __init__(self, md_df: pd.DataFrame, data_factors, init_cash=2e5, fee_rate=3e-3,
                 state_with_flag=False, reward_with_fee0=False, md_close_label='close', md_open_label='open'):
        self.data_close = md_df[md_close_label]
        self.data_open = md_df[md_open_label]
        self.data_observation = data_factors
        self.action_operations = ACTION_OPS
        self.fee_rate = fee_rate  # 千三手续费
        self.fee_curr_step = 0
        self.fee_tot = 0
        self.max_step_count = self.data_observation.shape[0] - 1
        self.init_cash = init_cash
        # reset use
        self.step_counter = 0
        self.cash = self.init_cash
        self.position = 0
        self.total_value = self.cash + self.position
        self.total_value_fee0 = self.cash + self.position
        self._flags = _FLAG_EMPTY
        self.state_with_flag = state_with_flag
        self.reward_with_fee0 = reward_with_fee0
        self.action_count = 0

    @property
    def flags(self):
        """外部访问 flag 标志位，为了方便 one_hot 模式，因此做 + 1 处理"""
        return self._flags + 1  # 转换成 one_hot 模式的 flag

    def reset(self):
        self.step_counter = 0
        self.cash = self.init_cash
        self.position = 0
        self.total_value = self.cash + self.position
        self.total_value_fee0 = self.cash + self.position
        self._flags = _FLAG_EMPTY
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

    def get_action_operations(self):
        return self.action_operations

    def long(self):
        self._flags = _FLAG_LONG
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.fee_rate)
        self.position = quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def short(self):
        self._flags = _FLAG_SHORT
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.fee_rate)
        self.position = - quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def keep(self):
        quotes = self.data_open[self.step_counter] * 10
        self.position = quotes * self._flags

    def close_long(self):
        self._flags = _FLAG_EMPTY
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.fee_rate)
        self.position = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def close_short(self):
        self._flags = _FLAG_EMPTY
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.fee_rate)
        self.position = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1

    def step(self, action: int):
        self.fee_curr_step = 0
        if action == ACTION_LONG:
            if self._flags == _FLAG_EMPTY:
                self.long()
            elif self._flags == _FLAG_SHORT:
                self.close_short()
                self.long()
            else:
                self.keep()

        elif action == ACTION_CLOSE:
            if self._flags == _FLAG_LONG:
                self.close_long()
            elif self._flags == _FLAG_SHORT:
                self.close_short()
            else:
                pass

        elif action == ACTION_SHORT:
            if self._flags == _FLAG_EMPTY:
                self.short()
            elif self._flags == _FLAG_LONG:
                self.close_long()
                self.short()
            else:
                self.keep()
        elif action == ACTION_KEEP:
            self.keep()
        else:
            raise ValueError(f"action should be one of keys {ACTIONS}")

        # 计算费用
        self.fee_tot += self.fee_curr_step
        self.total_value_fee0 = self.total_value + self.fee_tot

        # 计算价值
        price = self.data_close[self.step_counter]
        position = price * 10 * self._flags
        reward = self.cash + position - self.total_value
        self.step_counter += 1
        self.total_value = position + self.cash
        next_observation = self.data_observation[self.step_counter]

        done = False
        if self.total_value < price:
            done = True
        if self.step_counter >= self.max_step_count:
            done = True

        ret_state = (next_observation, np.array([self.flags])) if self.state_with_flag else next_observation
        ret_reward = (reward / price, (reward + self.fee_curr_step) / price) if self.reward_with_fee0 else (
                reward / price)

        return ret_state, ret_reward, done


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
    assert next_observation[1] == FLAG_EMPTY
    next_observation, reward, done = qm.step(ACTION_LONG)
    assert len(next_observation) == 2
    assert next_observation[1] == FLAG_LONG
    assert not done
    next_observation, reward, done = qm.step(ACTION_CLOSE)
    assert next_observation[1] == FLAG_EMPTY
    assert reward != 0
    next_observation, reward, done = qm.step(ACTION_CLOSE)
    assert next_observation[1] == FLAG_EMPTY
    assert reward == 0
    next_observation, reward, done = qm.step(ACTION_KEEP)
    assert next_observation[1] == FLAG_EMPTY
    assert reward == 0
    next_observation, reward, done = qm.step(ACTION_SHORT)
    assert next_observation[1] == FLAG_SHORT
    assert not done
    next_observation, reward, done = qm.step(ACTION_KEEP)
    assert next_observation[1] == FLAG_SHORT
    assert reward != 0
    try:
        qm.step(4)
    except ValueError:
        print('is ok for not supporting action>3')


if __name__ == "__main__":
    _test_quote_market()
