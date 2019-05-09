#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/4/14 12:23
@File    : plot2_test.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import unittest
from ibats_common.analysis.plot_db import show_order, show_trade


class SomeTest(unittest.TestCase):  # 继承unittest.TestCase
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
        print('set up')
        from ibats_common.example.ma_cross_stg import _test_use
        cls.stg_run_id = _test_use()

    def test_func(self):
        self.assertEqual(1, 1)

    def test_order_func(self):
        data_dict = show_order(stg_run_id=self.stg_run_id)
        self.assertEqual(len(data_dict), 1)
        key = list(data_dict.keys())[0]
        self.assertEqual(len(data_dict[key]['md']), 1)
        self.assertEqual(len(data_dict[key]['long_open_or_short_close']), 1)
        self.assertEqual(len(data_dict[key]['short_open_or_long_close']), 1)
        self.assertGreater(len(data_dict[key]['buy_sell_point_pair']), 2)
        self.assertGreater(len(data_dict[key]['sell_buy_point_pair']), 2)

    def test_trade_func(self):
        data_dict = show_trade(stg_run_id=self.stg_run_id)
        self.assertEqual(len(data_dict), 1)
        key = list(data_dict.keys())[0]
        self.assertEqual(len(data_dict[key]['md']), 1)
        self.assertEqual(len(data_dict[key]['long_open_or_short_close']), 1)
        self.assertEqual(len(data_dict[key]['short_open_or_long_close']), 1)
        self.assertGreater(len(data_dict[key]['buy_sell_point_pair']), 2)
        self.assertGreater(len(data_dict[key]['sell_buy_point_pair']), 2)


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
