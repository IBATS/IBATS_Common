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
from ibats_common.test.orm_trade_agent_detail_test import TradeAgentStatusDetailTest


class StgRunStatusDetailTest(unittest.TestCase):  # 继承unittest.TestCase
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

    def test_create_by_trade_agent_status_detail_list(self):
        info, status = TradeAgentStatusDetailTest.add_trade_agent_status_detail()
        detail = StgRunStatusDetail.create_by_trade_agent_status_detail_list(info.stg_run_id, [status])
        self.assertEqual(detail.stg_run_id, info.stg_run_id)
        self.assertEqual(detail.cash_available, status.cash_available)
        self.assertEqual(detail.position_value, status.position_value)
        self.assertEqual(detail.curr_margin, status.curr_margin)
        self.assertEqual(detail.close_profit, status.close_profit)
        self.assertEqual(detail.position_profit, status.position_profit)
        self.assertEqual(detail.floating_pl_cum, status.floating_pl_cum)
        self.assertEqual(detail.commission_tot, status.commission_tot)
        self.assertEqual(detail.cash_init, status.cash_init)
        self.assertEqual(detail.cash_and_margin, status.cash_and_margin)
        self.assertEqual(detail.cashflow, status.cashflow)
        self.assertEqual(detail.cashflow_cum, status.cashflow_cum)
        self.assertEqual(detail.rr, status.rr)


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
