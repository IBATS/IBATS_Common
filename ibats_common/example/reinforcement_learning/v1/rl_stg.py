#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:42
@File    : ma_cross_stg.py
@contact : mmmaaaggg@163.com
@desc    : 通过Q-Learn 算法。state：前5日收益率为 reward：次日收益率，每5个交易日进行一次重新训练
"""
from ibats_utils.mess import load_class

from ibats_common.common import BacktestTradeMode, ContextKey, Direction, CalcMode
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
    from ibats_common import module_root_path
    import os
    # 参数设置
    instrument_type = 'RB'
    run_mode = RunMode.Backtest_FixPercent
    calc_mode = CalcMode.Normal
    if retrain_period == 0:
        strategy_params = {'unit': 1,
                           'module_name': 'ibats_common.example.reinforcement_learning.v1.q_learn',
                           'class_name': 'RLHandler',
                           'q_table_key': q_table_key}
    else:
        strategy_params = {'unit': 1,
                           'module_name': 'ibats_common.example.reinforcement_learning.v1.q_learn',
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
    return stghandler


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
