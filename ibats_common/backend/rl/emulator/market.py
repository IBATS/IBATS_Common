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
from ibats_common.backend.rl.emulator.env import high2low, fix_data, get_factors


class QuotesMarket(object):
    def __init__(self, md_df):
        self.data_close = md_df['close']
        self.data_open = md_df['open']
        self.data_observation = DATA_FAC
        self.action_space = ['long', 'short', 'close']
        self.free = 1e-4

    def reset(self):
        self.step_counter = 0
        self.cash = 1e5
        self.position = 0
        self.total_value = self.cash + self.position
        self.flags = 0
        return self.data_observation[0]

    def get_action_space(self):
        return self.action_space

    def long(self):
        self.flags = 1
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.free)
        self.position = quotes

    def short(self):
        self.flags = -1
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.free)
        self.position = - quotes

    def keep(self):
        quotes = self.data_open[self.step_counter] * 10
        self.position = quotes * self.flags

    def close_long(self):
        self.flags = 0
        quotes = self.data_open[self.step_counter] * 10
        self.cash += quotes * (1 - self.free)
        self.position = 0

    def close_short(self):
        self.flags = 0
        quotes = self.data_open[self.step_counter] * 10
        self.cash -= quotes * (1 + self.free)
        self.position = 0

    def step_op(self, action):

        if action == 'long':
            if self.flags == 0:
                self.long()
            elif self.flags == -1:
                self.close_short()
                self.long()
            else:
                self.keep()

        elif action == 'close':
            if self.flags == 1:
                self.close_long()
            elif self.flags == -1:
                self.close_short()
            else:
                pass

        elif action == 'short':
            if self.flags == 0:
                self.short()
            elif self.flags == 1:
                self.close_long()
                self.short()
            else:
                self.keep()
        else:
            raise ValueError("action should be elements of ['long', 'short', 'close']")

        position = self.data_close[self.step_counter] * 10 * self.flags
        reward = self.cash + position - self.total_value
        self.step_counter += 1
        self.total_value = position + self.cash
        next_observation = self.data_observation[self.step_counter]

        done = False
        if self.total_value < 4000:
            done = True
        if self.step_counter > 1203:
            done = True

        return next_observation, reward, done

    def step(self, action):
        if action == 0:
            return self.step_op('long')
        elif action == 1:
            return self.step_op('short')
        elif action == 2:
            return self.step_op('close')
        else:
            raise ValueError("action should be one of [0,1,2]")


if __name__ == "__main__":
    pass
