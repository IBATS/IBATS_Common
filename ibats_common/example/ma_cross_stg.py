#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:42
@File    : ma_cross_stg.py
@contact : mmmaaaggg@163.com
@desc    : 简单的 MA5、MA10金叉、死叉多空策略，仅供测试及演示使用
"""
from ibats_common.common import BacktestTradeMode, ContextKey, Direction, CalcMode
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.md_agent import *
from ibats_local_trader.agent.td_agent import *

logger = logging.getLogger(__name__)


class MACrossStg(StgBase):

    def __init__(self, unit=1):
        super().__init__()
        self.unit = unit
        self.ma5 = []
        self.ma10 = []

    def on_prepare_min1(self, md_df, context):
        if md_df is not None:
            self.ma5 = list(md_df['close'].rolling(5, 5).mean())[10:]
            self.ma10 = list(md_df['close'].rolling(10, 10).mean())[10:]

    def on_min1(self, md_df, context):
        close = md_df['close'].iloc[-1]
        self.ma5.append(md_df['close'].iloc[-5:].mean())
        self.ma10.append(md_df['close'].iloc[-10:].mean())
        instrument_id = context[ContextKey.instrument_id_list][0]
        if self.ma5[-2] < self.ma10[-2] and self.ma5[-1] > self.ma10[-1]:
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
        elif self.ma5[-2] > self.ma10[-2] and self.ma5[-1] < self.ma10[-1]:
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


def _test_use(is_plot):
    from ibats_common.backend.mess import get_folder_path
    import os
    # 参数设置
    run_mode = RunMode.Backtest
    trade_mode = BacktestTradeMode.Order_2_Deal
    calc_mode = CalcMode.Normal
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': ['RB'],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2010-1-1',
        'file_path': os.path.abspath(os.path.join(
            get_folder_path('example', create_if_not_found=False), 'data', 'RB.csv')),
        'symbol_key': 'instrument_type',
    }]
    if run_mode == RunMode.Realtime:
        strategy_params = {'unit': 100}
        trade_agent_params = {
        }
        strategy_handler_param = {
        }
    elif run_mode == RunMode.Backtest:
        strategy_params = {'unit': 100}
        trade_agent_params = {
            'trade_mode': trade_mode,
            'init_cash': 1000000,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    else:
        # RunMode.Backtest_FixPercent
        strategy_params = {'unit': 1}
        trade_agent_params = {
            'trade_mode': trade_mode,
            "calc_mode": calc_mode,
        }
        strategy_handler_param = {
            'date_from': '2010-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }

    # 初始化策略处理器
    stghandler = strategy_handler_factory(
        stg_class=MACrossStg,
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
    is_plot = True
    _test_use(is_plot)
