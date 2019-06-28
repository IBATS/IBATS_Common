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
from ibats_utils.mess import date_2_str, str_2_date, load_class

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
        from ibats_common import module_root_path
        # ibats_common/example/module_data
        folder_path = os.path.join(module_root_path, 'example', 'module_data', 'v1')
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


class RLHandler:
    """
    RLHandler 是对 QLearningTable 的逻辑封装
    在策略使用是，可以方便的进行初始化及调用（将初始化，训练、获取 state、reward 等操作进行的同一封装）
    从而简化了 RLStg 策略中的代码复杂度
    """

    def __init__(self, q_table_key=None):
        """
        :param q_table_key:
        """
        action_space = ['empty', 'hold_long', 'hold_short']
        self.actions = list(range(len(action_space)))
        self.q_table_key = q_table_key
        self.ql_table = None
        self.last_state = None
        self.last_action = None
        self.enable_load_if_exist = True

    def init_state(self, md_df: pd.DataFrame):
        # 每一次新 episode 需要重置 state, action
        self.last_state = None
        self.last_action = None

        _ = self.choose_action(md_df)

    def init_ql_table(self, trade_date_to=None):
        key = self.q_table_key if self.q_table_key is not None else date_2_str(trade_date_to)
        self.ql_table = QLearningTable(actions=self.actions, key=key)

    def get_state_reward(self, md_df: pd.DataFrame):
        close_s = md_df['close']
        close_log_arr = np.log(close_s.iloc[-6:])
        close_latest = close_log_arr.iloc[-1]
        state = int((close_latest - close_log_arr.iloc[0]) * 1000)
        if self.last_action is not None:
            if self.last_action == 0:
                reward = 0
            elif self.last_action == 1:
                reward = int((close_latest - close_log_arr.iloc[-2]) * 1000)
            elif self.last_action == 2:
                reward = - int((close_latest - close_log_arr.iloc[-2]) * 1000)
            else:
                raise ValueError(f'last_action = {self.last_action} 值不合法')
        else:
            reward = 0

        return state, reward

    def choose_action(self, md_df: pd.DataFrame):
        trade_date_to = pd.to_datetime(md_df['trade_date'].iloc[-1])
        if self.ql_table is None:
            self.init_ql_table(trade_date_to)

        state, reward = self.get_state_reward(md_df)
        if self.last_state is not None:
            self.ql_table.learn(self.last_state, self.last_action, reward, state)

        self.last_action = self.ql_table.choose_action(state)
        self.last_state = state
        return self.last_action


class RLHandler4Train(RLHandler):
    """
    RLHandler 是对 QLearningTable 的逻辑封装
    在策略使用是，可以方便的进行初始化及调用（将初始化，训练、获取 state、reward 等操作进行的同一封装）
    从而简化了 RLStg 策略中的代码复杂度
    """

    def __init__(self, retrain_period, get_stg_handler, episode_count=3, q_table_key=None):
        """

        :param retrain_period: 0 不进行训练，int > 0 每隔 retrain_period 天重新训练
        :param episode_count:
        :param get_stg_handler: 用于构建 stg_handler 的函数，仅当 retrain_period > 0 时才会有用
        :param q_table_key:
        """
        RLHandler.__init__(self, q_table_key=q_table_key)
        self.retrain_period = pd.Timedelta(days=retrain_period)
        if get_stg_handler is None:
            raise ValueError('get_stg_handler must be not None')
        self.get_stg_handler = get_stg_handler
        self.do_train = retrain_period > 0 and get_stg_handler is not None
        self.train_date_latest = None
        self.train_date_from = None
        self.episode_count = episode_count

    def init_state(self, md_df: pd.DataFrame):
        # 每一次新 episode 需要重置 state, action
        self.last_state = None
        self.last_action = None
        if self.do_train:
            # for_train == False 当期为策略运行使用，在 on_prepare 阶段以及 on_period 定期进行重新训练
            trade_date_s = md_df['trade_date']
            self.train_date_from = pd.to_datetime(trade_date_s.iloc[0]) + self.retrain_period
            trade_date_to = trade_date_s.iloc[-1]
            self.train(date_2_str(trade_date_to))

        _ = self.choose_action(md_df)

    def train(self, trade_date_to):
        """
        具体功能参见 ibats_common.example.reinforcement_learning.v1.q_learn import main
        :param trade_date_to:
        :return:
        """
        if self.enable_load_if_exist:
            self.init_ql_table(trade_date_to)
            is_loaded = self.ql_table.load()
        else:
            is_loaded = False

        if self.ql_table is None:
            self.init_ql_table(trade_date_to)

        if not is_loaded:
            logger.info('开始训练：[%s, %s]', self.train_date_from, trade_date_to)
            env = Env(self.train_date_from, trade_date_to, self.get_stg_handler, q_table_key=self.ql_table.key)
            for episode in range(self.episode_count):
                # fresh env
                env.reset_and_start()
                # logger.info("\n%s", self.q_table.q_table)

            # end of game
            env.destroy()
            logger.info('RL Over')
            self.ql_table.save()
        # 设置最新训练日期
        self.train_date_latest = pd.to_datetime(trade_date_to)

    def choose_action(self, md_df: pd.DataFrame):
        trade_date_to = pd.to_datetime(md_df['trade_date'].iloc[-1])
        if self.do_train and trade_date_to > (self.train_date_latest + self.retrain_period):
            # for_train == False 当期为策略运行使用，在 on_prepare 阶段以及 on_period 定期进行重新训练
            self.train(date_2_str(trade_date_to))
        elif self.ql_table is None:
            self.init_ql_table(trade_date_to)

        state, reward = self.get_state_reward(md_df)
        if self.last_state is not None:
            self.ql_table.learn(self.last_state, self.last_action, reward, state)

        self.last_action = self.ql_table.choose_action(state)
        self.last_state = state
        return self.last_action


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
    from ibats_common.example.reinforcement_learning.v1.rl_stg import get_stg_handler
    env = Env(trade_date_from, trade_date_to, get_stg_handler, q_table_key=trade_date_to)
    action_space = ['empty', 'hold_long', 'hold_short']
    actions = list(range(len(action_space)))
    q_learn_class = load_class(module_name='ibats_common.example.reinforcement_learning.v1.q_learn',
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


def _test_rl_handler(trade_date_from='2010-1-1', trade_date_to='2018-10-18'):
    from ibats_common.example.data import load_data
    trade_date_from_ = str_2_date(trade_date_from)
    trade_date_to_ = str_2_date(trade_date_to)
    df = load_data('RB.csv')
    md_df = df[df['trade_date'].apply(lambda x: str_2_date(x) <= trade_date_to_)]
    trade_date_s = md_df['trade_date'].apply(lambda x: str_2_date(x))
    rl_handler = RLHandler(q_table_key=trade_date_to)
    rl_handler.init_state(md_df)
    trade_date_action_dic = {}
    for trade_date in trade_date_s[trade_date_s >= trade_date_from_]:
        md_df_curr = md_df[trade_date_s.apply(lambda x: x <= trade_date)]
        action = rl_handler.choose_action(md_df_curr)
        trade_date_action_dic[trade_date] = action

    action_df = pd.DataFrame([trade_date_action_dic]).T
    print("action_df = \n", action_df)
    ql_table = QLearningTable(actions=rl_handler.actions, key=trade_date_to)
    # assert ql_table.q_table.shape[0] == 0
    # ql_table_class = load_class(module_name='ibats_common.example.reinforcement_learning.q_learn',
    #                            class_name='QLearningTable')
    # ql_table = ql_table_class(actions=rl_handler.actions, key=trade_date_to)
    assert ql_table.q_table.shape[0] > 0
    print("ql_table.q_table.shape=", ql_table.q_table.shape)
    assert ql_table.q_table.shape == rl_handler.ql_table.q_table.shape
    # print(ql_table.q_table)


def _test_rl_handler_4_train(trade_date_from='2010-1-1', trade_date_to='2018-10-18'):
    from ibats_common.example.reinforcement_learning.v1.rl_stg import get_stg_handler
    from ibats_common.example.data import load_data
    trade_date_from_ = str_2_date(trade_date_from)
    trade_date_to_ = str_2_date(trade_date_to)
    df = load_data('RB.csv')
    md_df = df[df['trade_date'].apply(lambda x: str_2_date(x) <= trade_date_to_)]
    trade_date_s = md_df['trade_date'].apply(lambda x: str_2_date(x))
    rl_handler = RLHandler4Train(retrain_period=360, get_stg_handler=get_stg_handler, q_table_key=trade_date_to)
    rl_handler.init_state(md_df)
    trade_date_action_dic = {}
    for trade_date in trade_date_s[trade_date_s >= trade_date_from_]:
        md_df_curr = md_df[trade_date_s.apply(lambda x: x <= trade_date)]
        action = rl_handler.choose_action(md_df_curr)
        trade_date_action_dic[trade_date] = action

    action_df = pd.DataFrame([trade_date_action_dic]).T
    print("action_df = \n", action_df)
    # ql_table = QLearningTable(rl_handler.actions, key=trade_date_to)
    ql_table_class = load_class(module_name='ibats_common.example.reinforcement_learning.v1.q_learn',
                                class_name='QLearningTable')
    ql_table = ql_table_class(actions=rl_handler.actions, key=trade_date_to)
    assert ql_table.q_table.shape[0] > 0
    print("ql_table.q_table.shape=", ql_table.q_table.shape)
    # rl_handler.init_ql_table()
    assert ql_table.q_table.shape == rl_handler.ql_table.q_table.shape
    # print(ql_table.q_table)


if __name__ == "__main__":
    # _test_env()
    # _test_rl_handler()
    _test_rl_handler_4_train()
