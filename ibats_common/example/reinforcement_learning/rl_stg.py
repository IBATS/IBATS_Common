#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:42
@File    : ma_cross_stg.py
@contact : mmmaaaggg@163.com
@desc    : 简单的 MA5、MA10金叉、死叉多空策略，仅供测试及演示使用（回测模式：固定仓位比例）
"""
from ibats_utils.mess import load_class

from ibats_common.common import BacktestTradeMode, ContextKey, Direction, CalcMode
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *

logger = logging.getLogger(__name__)


class RLStg(StgBase):

    def __init__(self, module_name, class_name, for_train, unit=1):
        super().__init__()
        self.unit = unit
        self.state = None
        self.module_name, self.class_name = module_name, class_name
        self.for_train = for_train      # True 代表当期实例供RL训练使用，否则，用于策略模拟使用
        self.rl_handler_class = load_class(module_name, class_name)
        self.rl_handler = self.rl_handler_class()
        self.train_date_latest = None
        self.retrain_period = pd.Timedelta(days=10)
        self.train_date_from = None

    def on_prepare_min1(self, md_df, context):
        self.rl_handler.init_state()
        if not self.for_train:
            # for_train == False 当期为策略运行使用，在 on_prepare 阶段以及 on_period 定期进行重新训练
            trade_date_s = md_df['trade_date']
            self.train_date_from = trade_date_s.iloc[0] + self.retrain_period
            trade_date_to = trade_date_s.iloc[-1]
            train(self.rl_handler_class, self.train_date_from, trade_date_to, episode_count=3)
            self.train_date_latest = trade_date_to

        state, _ = self.rl_handler.get_state_reward(md_df)
        _ = self.rl_handler.choose_action(state)

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
        trade_date_s = md_df['trade_date']
        trade_date_to = trade_date_s.iloc[-1]
        if (not self.for_train) and trade_date_to > (self.train_date_latest + self.retrain_period):
            # for_train == False 当期为策略运行使用，在 on_prepare 阶段以及 on_period 定期进行重新训练
            train(self.rl_handler_class, self.train_date_from, trade_date_to, episode_count=3)

        close = md_df['close'].iloc[-1]
        state, reward = self.rl_handler.get_state_reward(md_df)
        self.rl_handler.learn(reward, state)
        action = self.rl_handler.choose_action(state)

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


def train(rl_handler_class, trade_date_from, trade_date_to, episode_count):
    """
    具体功能参见 ibats_common.example.reinforcement_learning.q_learn import main
    :param rl_handler_class:
    :param trade_date_from:
    :param trade_date_to:
    :param episode_count:
    :return:
    """
    from ibats_common.example.reinforcement_learning.q_learn import Env
    logger.info('开始训练：[%s, %s]', trade_date_from, trade_date_to)
    env = Env(trade_date_from, trade_date_to)
    rl_handler = rl_handler_class()
    for episode in range(episode_count):
        # fresh env
        env.reset()
        logger.info("\n%s", rl_handler.q_table.q_table)

    # end of game
    env.destroy()
    logger.info('RL Over')


def _test_use(is_plot):
    from ibats_common import module_root_path
    import os
    # 参数设置
    instrument_type = 'RB'
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    strategy_params = {'unit': 1,
                       'module_name': 'ibats_common.example.reinforcement_learning.q_learn',
                       'class_name': 'RLHandler',
                       'for_train': False}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': [instrument_type],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2010-1-1',
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
