#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-6-17 下午1:57
@File    : q_learn.py
@contact : mmmaaaggg@163.com
@desc    : Q Learning Table and Runner
"""

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from ibats_utils.mess import date_2_str, load_class
from ibats_common.example.rl_stg.v3 import module_version

logger = logging.getLogger(__name__)


def try_hash_or_str(v):
    try:
        if isinstance(v, dict):
            keys = list(v.keys())
            keys.sort()
            ret_str = "{" + ",".join([f"{try_hash_or_str(k)}:{try_hash_or_str(v[k])}" for k in keys]) + "}"
        elif isinstance(v, set):
            ret_str = "{" + ",".join([f"{try_hash_or_str(k)}" for k in v]) + "}"
        elif isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            ret_str = v
        elif isinstance(v, Iterable):
            ret_str = "[" + ",".join([f"{try_hash_or_str(k)}" for k in v]) + "]"
        else:
            ret_str = hash(v)

        return ret_str
    except TypeError:
        return str(v)


def singleton(cls):
    instance = {}

    def _singleton(*args, **kw):
        keys = list(kw.keys())
        keys.sort()
        key = (cls, try_hash_or_str(args), try_hash_or_str(kw))
        if key not in instance:
            logger.debug("单例 %s", key)
            instance[key] = cls(*args, **kw)
        else:
            logger.debug("单例 已存在%s", key)
        return instance[key]

    return _singleton


@singleton
class QLearningTable:
    """
    Q-Learn 的实现类，完成获取state、reward、learn 等各种操作
    QLearningTable 采用的是带参数单利模式，
    因此，通过创建同样参数的实例，即可获取的在Env环境下训练后的 QLearningTable 结果
    """

    def __init__(self, actions, key=None, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.key = key  # 对于每一个Q table 的标识同样参数的情况下，区别不同q table使用
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def _get_file_path(self, key):
        from ibats_common.backend.mess import get_folder_path
        folder_path = os.path.join(get_folder_path('example', create_if_not_found=False), 'module_data', module_version)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = f"q_table_{key}_{len(self.actions)}.csv"
        file_path = os.path.join(folder_path, file_name)
        return file_path

    def save(self):
        file_path = self._get_file_path(self.key)
        self.q_table.to_csv(file_path, index=False)

    def load(self):
        file_path = self._get_file_path(self.key)
        if os.path.exists(file_path):
            self.q_table = pd.read_csv(file_path)
            self.q_table.columns = self.q_table.columns.astype(np.int)
            return True
        else:
            return False

    def reset(self):
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


class Env:
    """
    Env 用于构建一个策略模拟环境，在该环境下利用 get_stg_handler 产生相应的 stg_handler 循环执行，
    以便对 QLearningTable 进行循环训练，
    QLearningTable 采用的是带参数单利模式，
    因此，通过创建同样参数的实例，即可获取的在Env环境下训练后的 QLearningTable 结果
    """

    def __init__(self, date_from, date_to, get_stg_handler, q_table_key=None):
        self.action_space = ['empty', 'hold_long', 'hold_short']
        self.n_actions = len(self.action_space)
        self.q_table_key = q_table_key
        self.stg_handler = None
        self.train_date_from, self.train_date_to = date_2_str(date_from), date_2_str(date_to)
        self.get_stg_handler = get_stg_handler
        self._build()

    def _build(self):
        # 初始化策略处理器
        get_stg_handler = self.get_stg_handler
        self.stg_handler = stg_handler = get_stg_handler(retrain_period=0, q_table_key=self.q_table_key)
        return stg_handler

    def set_stop(self):
        if self.stg_handler is not None:
            stg_run_id = self.stg_handler.stg_run_id
            if self.stg_handler.is_alive():
                logger.info('结束旧任务 stg_run_id=%d ...', stg_run_id)
                self.stg_handler.keep_running = False
                self.stg_handler.join()
                logger.info('结束旧任务 stg_run_id=%d 完成', stg_run_id)
            # 输出绩效报告
            # from ibats_common.analysis.summary import summary_stg_2_docx
            # from ibats_utils.mess import open_file_with_system_app
            # file_path = summary_stg_2_docx(stg_run_id, enable_clean_cache=False)
            # if file_path is not None:
            #     open_file_with_system_app(file_path)

    def reset_and_start(self):
        self.set_stop()
        stg_handler = self._build()
        stg_run_id = stg_handler.stg_run_id
        stg_handler.start()
        logging.info("执行开始 stg_run_id = %d", stg_run_id)

    def destroy(self):
        self.set_stop()


def _test_env(trade_date_from='2010-1-1', trade_date_to='2018-10-18'):
    from ibats_common.example.rl_stg.v3.rl_stg import get_stg_handler
    env = Env(trade_date_from, trade_date_to, get_stg_handler, q_table_key=trade_date_to)
    action_space = ['empty', 'hold_long', 'hold_short']
    actions = list(range(len(action_space)))
    q_learn_class = load_class(module_name=f'ibats_common.example.reinforcement_learning.{module_version}.q_learn',
                               class_name='QLearningTable')
    q_learn = q_learn_class(actions=actions, key=trade_date_to)
    for episode in range(2):
        # fresh env
        env.reset_and_start()
        # logger.info("q_learn.q_table\n%s", q_learn.q_table)
        print('q_learn.q_table')
        print(q_learn.q_table)

    # end of game
    env.destroy()
    logger.info('RL Over')


if __name__ == "__main__":
    _test_env()
