#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/3/8 12:23
@File    : orm_stg_run_status_detail_test.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_common.backend.orm import *
import unittest
from ibats_common.test.orm_init_test import InitTest


class PosStatusDetailTest(unittest.TestCase):  # 继承unittest.TestCase
    def tearDown(self):
        pass

    def setUp(self):
        # 每个测试用例执行之前做操作
        pass

    @classmethod
    def tearDownClass(cls):
        # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        pass

    @classmethod
    def setUpClass(cls):
        # from ibats_common.config import update_db_config, ConfigBase
        # update_db_config({
        #     ConfigBase.DB_SCHEMA_IBATS: 'mysql://mg:Dcba1234@localhost/' + ConfigBase.DB_SCHEMA_IBATS,
        # })
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        init()
        global engine_ibats
        engine_ibats = engines.engine_ibats

    def test_func(self):
        pass

    def test_create_by_trade_detail(self):
        trade = PosStatusDetailTest.add_trade()
        status = PosStatusDetail.create_by_trade_detail(trade)
        self.assertIsInstance(status, PosStatusDetail)
        self.assertEqual(status.direction, trade.direction)
        self.assertEqual(status.symbol, trade.symbol)
        self.assertEqual(status.position, trade.trade_vol)
        self.assertEqual(status.position_chg, trade.trade_vol)
        self.assertEqual(status.margin, trade.margin)
        self.assertEqual(status.margin_chg, trade.margin)
        self.assertEqual(status.floating_pl // 1, -trade.commission // 1)
        self.assertEqual(status.floating_pl_rate, -trade.commission / status.margin)  # 初次建仓时浮动收益就等于手续费 / 保证金
        self.assertEqual(status.floating_pl_chg, status.floating_pl)
        self.assertEqual(status.floating_pl_cum, status.floating_pl)
        self.assertEqual(status.cashflow, status.cashflow)
        self.assertEqual(status.cashflow_cum, status.cashflow_cum)
        self.assertEqual(status.calc_mode, status.calc_mode)
        self.assertEqual(status.rr, status.floating_pl_rate)
        self.assertEqual(status.commission, trade.commission)
        self.assertEqual(status.position_date_type, PositionDateType.Today.value)
        self.assertEqual(status.multiple, trade.multiple)
        self.assertEqual(status.margin_ratio, trade.margin_ratio)

    def test_create_self(self):
        trade = PosStatusDetailTest.add_trade()
        status = PosStatusDetail.create_by_trade_detail(trade)
        status2 = status.create_by_self()
        self.assertIsInstance(status2, PosStatusDetail)
        self.assertEqual(status2.direction, status.direction)
        self.assertEqual(status2.symbol, status.symbol)
        self.assertEqual(status2.position, status.position)
        self.assertEqual(status2.position_chg, 0)
        self.assertEqual(status2.margin, status.margin)
        self.assertEqual(status2.margin_chg, 0)
        self.assertEqual(status2.floating_pl, status.floating_pl)
        self.assertEqual(status2.floating_pl_rate, 0)
        self.assertEqual(status2.floating_pl_chg, 0)
        self.assertEqual(status2.floating_pl_cum, status.floating_pl_cum)
        self.assertEqual(status2.cashflow, 0)
        self.assertEqual(status2.cashflow_cum, status.cashflow_cum)
        self.assertEqual(status2.calc_mode, status.calc_mode)
        self.assertEqual(status2.rr, 0)
        self.assertEqual(status2.commission, 0)
        self.assertEqual(status2.position_date_type, status.position_date_type)
        self.assertEqual(status2.multiple, status.multiple)
        self.assertEqual(status2.margin_ratio, status.margin_ratio)

    def test_update_by_trade_detail1(self):
        # 多头加仓
        trade = PosStatusDetailTest.add_trade()
        status = PosStatusDetail.create_by_trade_detail(trade)
        trade2 = PosStatusDetailTest.add_trade_4_test_update_by_trade_detail1()
        self.assertEqual(trade.trade_vol * 2, trade2.trade_vol)
        status2 = status.update_by_trade_detail(trade2)
        self.assertIsInstance(status2, PosStatusDetail)
        self.assertEqual(status2.direction, trade.direction)
        self.assertEqual(status2.symbol, trade.symbol)
        self.assertEqual(status2.position, trade.trade_vol + trade2.trade_vol)
        self.assertEqual(status2.position_chg, trade2.trade_vol)
        self.assertEqual(status2.margin, trade.margin + trade2.margin)
        self.assertEqual(status2.margin_chg, trade2.margin)
        self.assertEqual(status2.floating_pl // 1, -(trade.commission + trade2.commission) // 1)
        # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status2.margin * 10000 // 1)
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
        self.assertEqual(status2.cashflow, - status2.margin_chg - status2.commission)
        self.assertEqual(status2.cashflow_cum, status.cashflow_cum + status2.cashflow)
        self.assertEqual(status2.calc_mode, status.calc_mode)
        self.assertEqual(status2.rr * 10000 // 1, status2.floating_pl_rate * 10000 // 1)
        self.assertEqual(status2.commission, trade2.commission)
        self.assertEqual(status2.commission_tot, trade.commission + trade2.commission)
        self.assertEqual(status2.position_date_type, PositionDateType.Today.value)
        self.assertEqual(status2.multiple, trade.multiple)
        self.assertEqual(status2.margin_ratio, trade.margin_ratio)

    def test_update_by_trade_detail2(self):
        # 多头减仓
        trade = PosStatusDetailTest.add_trade()
        status = PosStatusDetail.create_by_trade_detail(trade)
        trade2 = PosStatusDetailTest.add_trade_4_test_update_by_trade_detail2()
        self.assertEqual(trade2.trade_vol, trade.trade_vol / 2)
        status2 = status.update_by_trade_detail(trade2)
        self.assertIsInstance(status2, PosStatusDetail)
        self.assertEqual(status2.direction, trade.direction)
        self.assertEqual(status2.symbol, trade.symbol)
        self.assertEqual(status2.position, trade.trade_vol - trade2.trade_vol)
        self.assertEqual(status2.position_chg, -trade2.trade_vol)
        self.assertEqual(status2.margin, trade.margin - trade2.margin)
        self.assertEqual(status2.margin_chg, -trade2.margin)
        self.assertEqual(status2.floating_pl // 1, -(trade.commission + trade2.commission) // 1)
        # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status2.margin * 10000 // 1)
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
        self.assertEqual(status2.cashflow, - status2.margin_chg - status2.commission)
        self.assertEqual(status2.cashflow_cum, status.cashflow_cum + status2.cashflow)
        self.assertEqual(status2.calc_mode, status.calc_mode)
        self.assertEqual(status2.rr * 10000 // 1, status2.floating_pl_rate * 10000 // 1)
        self.assertEqual(status2.commission, trade2.commission)
        self.assertEqual(status2.commission_tot, trade.commission + trade2.commission)
        self.assertEqual(status2.position_date_type, PositionDateType.Today.value)
        self.assertEqual(status2.multiple, trade.multiple)
        self.assertEqual(status2.margin_ratio, trade.margin_ratio)

    def test_update_by_trade_detail3(self):
        # 多头平仓
        trade = PosStatusDetailTest.add_trade()
        status = PosStatusDetail.create_by_trade_detail(trade)
        trade2 = PosStatusDetailTest.add_trade_4_test_update_by_trade_detail3()
        self.assertEqual(trade2.trade_vol, trade.trade_vol)
        status2 = status.update_by_trade_detail(trade2)
        self.assertIsInstance(status2, PosStatusDetail)
        self.assertEqual(status2.direction, trade.direction)
        self.assertEqual(status2.symbol, trade.symbol)
        self.assertEqual(status2.position, 0)
        self.assertEqual(status2.position_chg, -trade2.trade_vol)
        self.assertEqual(status2.margin, 0)
        self.assertEqual(status2.margin_chg, -trade2.margin)
        self.assertEqual(status2.floating_pl // 1, -(trade.commission + trade2.commission) // 1)
        # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status.margin * 10000 // 1)
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
        self.assertEqual(status2.cashflow, - status2.margin_chg - status2.commission)
        self.assertEqual(status2.cashflow_cum, status.cashflow_cum + status2.cashflow)
        self.assertEqual(status2.calc_mode, status.calc_mode)
        self.assertEqual(status2.rr * 10000 // 1, status2.floating_pl_rate * 10000 // 1)
        self.assertEqual(status2.commission, trade2.commission)
        self.assertEqual(status2.commission_tot, trade.commission + trade2.commission)
        self.assertEqual(status2.position_date_type, PositionDateType.Today.value)
        self.assertEqual(status2.multiple, trade.multiple)
        self.assertEqual(status2.margin_ratio, trade.margin_ratio)

    @staticmethod
    def add_trade(stg_run_id=None):
        order = InitTest.add_order(stg_run_id)
        trade = TradeDetail.create_by_order_detail(order)
        return trade

    @staticmethod
    def add_trade_4_test_update_by_trade_detail1():
        order = InitTest.add_order()
        order.order_vol += order.order_vol
        trade = TradeDetail.create_by_order_detail(order)
        trade.trade_date += timedelta(days=1)
        trade.trade_dt += timedelta(days=1)
        return trade

    @staticmethod
    def add_trade_4_test_update_by_trade_detail2():
        order = InitTest.add_order()
        order.order_vol = order.order_vol / 2
        order.action = Action.Close.value
        trade = TradeDetail.create_by_order_detail(order)
        trade.trade_date += timedelta(days=1)
        trade.trade_dt += timedelta(days=1)
        return trade

    @staticmethod
    def add_trade_4_test_update_by_trade_detail3():
        order = InitTest.add_order()
        order.action = Action.Close.value
        trade = TradeDetail.create_by_order_detail(order)
        trade.trade_date += timedelta(days=1)
        trade.trade_dt += timedelta(days=1)
        return trade


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
