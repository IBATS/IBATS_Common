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
from datetime import date, datetime, time
import unittest
from ibats_common.test.orm_init_test import InitTest

from ibats_common.common import ExchangeName


class TradeDetailTest(unittest.TestCase):  # 继承unittest.TestCase
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

    def test_create_by_order_detail(self):
        order = InitTest.add_order()
        trade = TradeDetail.create_by_order_detail(order)
        self.assertIsInstance(trade, TradeDetail)
        self.assertGreater(trade.commission, 0)
        self.assertGreater(trade.margin, 0)
        self.assertGreater(trade.multiple, 0)
        self.assertGreater(trade.margin_ratio, 0)
        self.assertEqual(trade.trade_vol, order.order_vol)
        self.assertEqual(trade.trade_price, order.order_price)
        self.assertEqual(trade.direction, order.direction)
        self.assertEqual(trade.action, order.action)


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
