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

import numpy as np
import pandas as pd
from ibats_utils.mess import date_2_str

logger = logging.getLogger(__name__)


def singleton(cls, *args, **kw):
    instance = {}

    def _singleton():
        logger.info("单例 %s", cls)
        if cls not in instance:
            instance[cls] = cls(*args, **kw)
        return instance[cls]

    return _singleton


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
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
        import os
        from ibats_common import module_root_path
        # ibats_common/example/module_data
        folder_path = os.path.join(module_root_path, 'example', 'module_data')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = f"q_table_{key}_{len(self.actions)}.csv"
        file_path = os.path.join(folder_path, file_name)
        return file_path

    def save(self, key):
        file_path = self._get_file_path(key)
        self.q_table.to_csv(file_path)

    def load(self, key):
        file_path = self._get_file_path(key)
        self.q_table = pd.read_csv(file_path)

    def reset(self):
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


@singleton
class RLHandler:
    __instance = None

    def __init__(self):
        action_space = ['empty', 'hold_long', 'hold_short']
        self.q_table = QLearningTable(actions=list(range(len(action_space))))
        self.last_state = None
        self.last_action = None

    def init_state(self):
        # 每一次新 episode 需要重置 state, action
        self.last_state = None
        self.last_action = None

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
                raise ValueError('last_action = %d 值不合法', self.last_action)
        else:
            reward = 0

        return state, reward

    def choose_action(self, state):
        self.last_action = self.q_table.choose_action(state)
        self.last_state = state
        return self.last_action

    def learn(self, r, s_):
        if self.last_state is None:
            return
        self.q_table.learn(self.last_state, self.last_action, r, s_)


class Env:
    def __init__(self, date_from, date_to):
        self.action_space = ['empty', 'hold_long', 'hold_short']
        self.n_actions = len(self.action_space)
        self.stg_handler = None
        self.train_date_from, self.train_date_to = date_2_str(date_from), date_2_str(date_to)
        self.rl_handler = RLHandler()
        self._build()

    def _build(self):
        from ibats_common import module_root_path
        import os
        from ibats_common.common import RunMode
        from ibats_common.common import CalcMode
        from ibats_common.common import PeriodType
        from ibats_common.common import BacktestTradeMode
        from ibats_common.strategy_handler import strategy_handler_factory
        from ibats_common.common import ExchangeName
        from ibats_common.example.reinforcement_learning.rl_stg import RLStg
        # 参数设置
        instrument_type = 'RB'
        run_mode = RunMode.Backtest_FixPercent
        calc_mode = CalcMode.Normal
        strategy_params = {'unit': 1,
                           'module_name': 'ibats_common.example.reinforcement_learning.q_learn',
                           'class_name': 'RLHandler',
                           'for_train': True}
        md_agent_params_list = [{
            'md_period': PeriodType.Min1,
            'instrument_id_list': [instrument_type],
            'datetime_key': 'trade_date',
            'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
            'init_md_date_to': self.train_date_from,
            # 'C:\GitHub\IBATS_Common\ibats_common\example\ru_price2.csv'
            'file_path': os.path.abspath(os.path.join(module_root_path, 'example', 'data', 'RB.csv')),
            'symbol_key': 'instrument_type',
        }]
        if run_mode == RunMode.Realtime:
            trade_agent_params = {
            }
            strategy_handler_param = {
            }
        elif run_mode == RunMode.Backtest:
            trade_agent_params = {
                'trade_mode': BacktestTradeMode.Order_2_Deal,
                'init_cash': 1000000,
                "calc_mode": calc_mode,
            }
            strategy_handler_param = {
                'date_from': self.train_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
                'date_to': self.train_date_to,
            }
        else:
            # RunMode.Backtest_FixPercent
            trade_agent_params = {
                'trade_mode': BacktestTradeMode.Order_2_Deal,
                "calc_mode": calc_mode,
            }
            strategy_handler_param = {
                'date_from': self.train_date_from,  # 策略回测历史数据，回测指定时间段的历史行情
                'date_to': self.train_date_to,
            }

        # 初始化策略处理器
        self.stg_handler = stg_handler = strategy_handler_factory(
            stg_class=RLStg,
            strategy_params=strategy_params,
            md_agent_params_list=md_agent_params_list,
            exchange_name=ExchangeName.LocalFile,
            run_mode=run_mode,
            trade_agent_params=trade_agent_params,
            strategy_handler_param=strategy_handler_param,
        )
        stg_run_id = stg_handler.stg_run_id
        stg_handler.start()
        logging.info("执行开始 stg_run_id = %d", stg_run_id)
        # time.sleep(10)
        # stg_handler.keep_running = False
        # stg_handler.join()
        # logging.info("执行结束 stg_run_id = %d", stg_run_id)
        # if is_plot:
        #     from ibats_common.analysis.summary import summary_stg_2_docx
        #     from ibats_utils.mess import open_file_with_system_app
        #     file_path = summary_stg_2_docx(stg_run_id, enable_clean_cache=False)
        #     if file_path is not None:
        #         open_file_with_system_app(file_path)

    def set_stop(self):
        if self.stg_handler is not None:
            stg_run_id = self.stg_handler.stg_run_id
            self.stg_handler.keep_running = False
            logger.info('结束旧任务 stg_run_id=%d', stg_run_id)
            self.stg_handler.join()
            logger.info('结束旧任务 stg_run_id=%d 完成', stg_run_id)
            # 输出绩效报告
            # from ibats_common.analysis.summary import summary_stg_2_docx
            # from ibats_utils.mess import open_file_with_system_app
            # file_path = summary_stg_2_docx(stg_run_id, enable_clean_cache=False)
            # if file_path is not None:
            #     open_file_with_system_app(file_path)

    def reset(self):
        self.set_stop()
        self._build()

    def destroy(self):
        self.set_stop()


def main(trade_date_from='2010-1-1', trade_date_to='2018-10-18'):
    env = Env(trade_date_from, trade_date_to)
    rl_handler = RLHandler()
    for episode in range(2):
        # fresh env
        env.reset()
        logger.info("\n%s", rl_handler.q_table.q_table)

    # end of game
    env.destroy()
    logger.info('RL Over')


if __name__ == "__main__":
    main()
