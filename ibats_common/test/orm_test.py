#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/3/8 12:23
@File    : orm_test.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_common.backend.orm import *
import unittest

from ibats_common.common import ExchangeName


class MyTest(unittest.TestCase):  # 继承unittest.TestCase
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
        from ibats_common.config import update_db_config, ConfigBase
        update_db_config({
            ConfigBase.DB_SCHEMA_IBATS: 'mysql://mg:Dcba1234@localhost/' + ConfigBase.DB_SCHEMA_IBATS,
        })
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        init()
        global engine_ibats
        engine_ibats = engines.engine_ibats

    @staticmethod
    def add_stg_run_info():
        info = StgRunInfo()
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(info)
            session.commit()
        return info

    def test_add_stg_run_info(self):
        info = MyTest.add_stg_run_info()

        self.assertIsNotNone(info.stg_run_id)
        self.assertGreater(info.stg_run_id, -1)

        with with_db_session(engine_ibats) as session:
            info2 = session.query(StgRunInfo).filter(StgRunInfo.stg_run_id == info.stg_run_id).first()

        self.assertIsInstance(info2, StgRunInfo)
        self.assertEqual(info.stg_run_id, info2.stg_run_id)

    @staticmethod
    def add_order():
        info = MyTest.add_stg_run_info()
        order = OrderDetail(info.stg_run_id, trade_agent_key=ExchangeName.DataIntegration,
                            order_dt=datetime.now(), order_date=datetime.today(), order_time=datetime.now().time(),
                            order_millisec=99, direction=int(Direction.Long), action=int(Action.Open), symbol='RB1801',
                            order_price=1000.0, order_vol=20
                            )
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(order)
            session.commit()
        return order

    def test_add_order_detail(self):
        order = MyTest.add_order()
        self.assertIsNotNone(order)
        self.assertGreater(order.order_idx, -1)

        with with_db_session(engine_ibats) as session:
            order2 = session.query(OrderDetail).filter(OrderDetail.order_idx == order.order_idx).first()

        self.assertIsInstance(order2, OrderDetail)
        self.assertEqual(order2.order_idx, order.order_idx)

    def test_add_trade_detail(self):
        trade = MyTest.add_trade()

        self.assertIsNotNone(trade)
        self.assertGreater(trade.trade_idx, -1)

        with with_db_session(engine_ibats) as session:
            trade2 = session.query(TradeDetail).filter(TradeDetail.trade_idx == trade.trade_idx).first()

        self.assertIsInstance(trade2, TradeDetail)
        self.assertEqual(trade2.trade_idx, trade2.trade_idx)

    @staticmethod
    def add_trade():
        order = MyTest.add_order()
        trade = TradeDetail(order.stg_run_id, trade_agent_key=ExchangeName.DataIntegration, order_idx=order.order_idx,
                            order_price=1000.0, order_vol=20,
                            trade_dt=datetime.now(), trade_date=datetime.today(), trade_time=datetime.now().time(),
                            trade_millisec=99, direction=int(Direction.Long), action=int(Action.Open), symbol='RB1801',
                            trade_price=order.order_price + 1, trade_vol=order.order_vol,
                            margin=order.order_price * order.order_vol,
                            commission=order.order_price * order.order_vol + 0.00005, multiple=10, margin_ratio=1
                            )
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(trade)
            session.commit()
        return trade


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
