#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:42
@File    : ma_cross_stg.py
@contact : mmmaaaggg@163.com
@desc    : V2主要是在V1基础上进行了结构重整，将 q_learn中的部分代码转移到 rl_stg 中
"""
from ibats_utils.mess import load_class, date_2_str
import numpy as np
from ibats_common.common import BacktestTradeMode, ContextKey, Direction, CalcMode
from ibats_common.example.rl_stg.v3.q_learn import QLearningTable, Env
from ibats_common.example.rl_stg.v3 import module_version
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *

logger = logging.getLogger(__name__)


class RLStg(StgBase):

    def __init__(self, module_name, class_name, retrain_period=0, unit=1, q_table_key=None):
        """

        :param module_name:
        :param class_name:
        :param retrain_period: 0 不进行训练，int > 0 每隔 retrain_period 天重新训练
        :param unit:
        """
        super().__init__()
        self.unit = unit
        self.state = None
        self.module_name, self.class_name = module_name, class_name
        self.rl_handler_class = load_class(module_name, class_name)
        # RLHandler4Train 作为 RLHandler 的子类，完全支持 RLHandler 所有功能，这里将其区分开来
        # 主要是为了防止视觉上混淆，以及理解上的偏差
        if retrain_period == 0:
            # 无训练功能，仅进行买入卖出动作判断，供 Env 环境下训练时被调用
            self.rl_handler = self.rl_handler_class(q_table_key=q_table_key)
        else:
            # 有训练功能，将定期调用 RLHandler4Train 启动 Env 环境进行训练
            self.rl_handler = self.rl_handler_class(
                retrain_period=retrain_period, get_stg_handler=get_stg_handler, q_table_key=q_table_key)

    def on_prepare_min1(self, md_df, context):
        self.rl_handler.init_state(md_df)

    def long_position(self, close, instrument_id):
        position_date_pos_info_dic = self.get_position(instrument_id)
        no_holding_target_position = True
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Short:
                    self.close_short(instrument_id, close, pos_info.position)
                elif direction == Direction.Long:
                    no_holding_target_position = False
        if no_holding_target_position:
            self.open_long(instrument_id, close, self.unit)

    def short_position(self, close, instrument_id):
        position_date_pos_info_dic = self.get_position(instrument_id)
        no_holding_target_position = True
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Long:
                    self.close_long(instrument_id, close, pos_info.position)
                elif direction == Direction.Short:
                    no_holding_target_position = False
        if no_holding_target_position:
            self.open_short(instrument_id, close, self.unit)

    def empty_position(self, close, instrument_id):
        position_date_pos_info_dic = self.get_position(instrument_id)
        if position_date_pos_info_dic is not None:
            for position_date, pos_info in position_date_pos_info_dic.items():
                if pos_info.position == 0:
                    continue
                direction = pos_info.direction
                if direction == Direction.Long:
                    self.close_long(instrument_id, close, pos_info.position)
                elif direction == Direction.Short:
                    self.close_short(instrument_id, close, pos_info.position)

    def on_min1(self, md_df, context):
        close = md_df['close'].iloc[-1]
        action = self.rl_handler.choose_action(md_df)

        instrument_id = context[ContextKey.instrument_id_list][0]
        if action == 1:
            # 做多
            self.long_position(close, instrument_id)
        elif action == 2:
            # 做空
            self.short_position(close, instrument_id)
        elif action == 0:
            # 空仓
            self.empty_position(close, instrument_id)


def get_stg_handler(retrain_period, q_table_key=None):
    from ibats_common.backend.mess import get_folder_path
    import os
    # 参数设置
    instrument_type = 'RB'
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    if retrain_period == 0:
        strategy_params = {'unit': 1,
                           'module_name': f'ibats_common.example.rl_stg.{module_version}.rl_stg',
                           'class_name': 'RLHandler',
                           'q_table_key': q_table_key}
    else:
        strategy_params = {'unit': 1,
                           'module_name': f'ibats_common.example.rl_stg.{module_version}.rl_stg',
                           'class_name': 'RLHandler4Train',
                           'q_table_key': q_table_key,
                           'retrain_period': retrain_period
                           }

    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': [instrument_type],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2010-1-1',
        # 'C:\GitHub\IBATS_Common\ibats_common\example\ru_price2.csv'
        'file_path': os.path.abspath(os.path.join(
            get_folder_path('example', create_if_not_found=False), 'data', 'RB.csv')),
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
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    else:
        # RunMode.Backtest_FixPercent
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }

    # 初始化策略处理器
    stghandler = strategy_handler_factory(
        stg_class=RLStg,
        strategy_params=strategy_params,
        md_agent_params_list=md_agent_params_list,
        exchange_name=ExchangeName.LocalFile,
        run_mode=run_mode,
        trade_agent_params=trade_agent_params,
        strategy_handler_param=strategy_handler_param,
    )
    return stghandler


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
        具体功能参见 ibats_common.example.rl_stg.q_learn import main
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
    # ql_table_class = load_class(module_name='ibats_common.example.rl_stg.q_learn',
    #                            class_name='QLearningTable')
    # ql_table = ql_table_class(actions=rl_handler.actions, key=trade_date_to)
    assert ql_table.q_table.shape[0] > 0
    print("ql_table.q_table.shape=", ql_table.q_table.shape)
    assert ql_table.q_table.shape == rl_handler.ql_table.q_table.shape
    # print(ql_table.q_table)


def _test_rl_handler_4_train(trade_date_from='2010-1-1', trade_date_to='2018-10-18'):
    from ibats_common.example.rl_stg.v1.rl_stg import get_stg_handler
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
    ql_table_class = load_class(module_name=f'ibats_common.example.rl_stg.{module_version}.q_learn',
                                class_name='QLearningTable')
    ql_table = ql_table_class(actions=rl_handler.actions, key=trade_date_to)
    assert ql_table.q_table.shape[0] > 0
    print("ql_table.q_table.shape=", ql_table.q_table.shape)
    # rl_handler.init_ql_table()
    assert ql_table.q_table.shape == rl_handler.ql_table.q_table.shape
    # print(ql_table.q_table)


def _test_use(is_plot):
    stghandler = get_stg_handler(retrain_period=7)
    stghandler.start()
    time.sleep(10)
    stghandler.keep_running = False
    stghandler.join()
    stg_run_id = stghandler.stg_run_id
    logging.info("执行结束 stg_run_id = %d", stg_run_id)

    if is_plot:
        # from ibats_common.analysis.plot_db import show_order, show_cash_and_margin, show_rr_with_md
        # from ibats_common.analysis.summary import summary_rr
        # show_order(stg_run_id)
        # df = show_cash_and_margin(stg_run_id)
        # sum_df, symbol_rr_dic, save_file_path_dic = show_rr_with_md(stg_run_id)
        # for symbol, rr_df in symbol_rr_dic.items():
        #     col_transfer_dic = {'return': rr_df.columns}
        #     summary_rr(rr_df, figure_4_each_col=True, col_transfer_dic=col_transfer_dic)
        from ibats_common.analysis.summary import summary_stg_2_docx
        from ibats_utils.mess import open_file_with_system_app
        file_path = summary_stg_2_docx(stg_run_id)
        if file_path is not None:
            open_file_with_system_app(file_path)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    _test_use(is_plot=True)
    # _test_rl_handler()
    # _test_rl_handler_4_train()
