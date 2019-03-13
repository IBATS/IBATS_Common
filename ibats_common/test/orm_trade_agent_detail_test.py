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


class TradeAgentStatusDetailTest(unittest.TestCase):  # 继承unittest.TestCase
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

    def test_create(self):
        _, status = TradeAgentStatusDetailTest.add_trade_agent_status_detail()
        with with_db_session(engine_ibats) as session:
            status2 = session.query(TradeAgentStatusDetail).filter(
                TradeAgentStatusDetail.trade_agent_status_detail_idx == status.trade_agent_status_detail_idx,
                TradeAgentStatusDetail.stg_run_id == status.stg_run_id
            ).first()
        self.assertIsInstance(status2, TradeAgentStatusDetail)
        self.assertGreater(status2.trade_agent_status_detail_idx, -1)
        self.assertGreater(status2.cash_init, 0)
        self.assertEqual(status2.cash_init, status2.cash_available)
        self.assertEqual(status2.cash_init, status2.cash_and_margin)
        self.assertEqual(status2.commission_tot, 0)
        self.assertEqual(status2.floating_pl_cum, 0)
        self.assertEqual(status2.position_profit, 0)
        self.assertEqual(status2.close_profit, 0)
        self.assertEqual(status2.curr_margin, 0)
        self.assertIsInstance(status2.trade_dt, datetime)
        self.assertIsInstance(status2.trade_time, time)
        self.assertIsInstance(status2.trade_millisec, int)

    def test_update_by_pos_status_detail(self):
        info, status = TradeAgentStatusDetailTest.add_trade_agent_status_detail()
        pos_status = TradeAgentStatusDetailTest.add_pos_status_detail(info)
        md = {
            'ActionDay': '2018-12-15',
            'ActionTime': '13:24:35',
        }
        status2 = status.update_by_pos_status_detail({pos_status.symbol: pos_status}, md=md)
        self.assertIsInstance(status2, TradeAgentStatusDetail)
        self.assertEqual(status2.stg_run_id, info.stg_run_id)
        self.assertGreater(status2.trade_agent_status_detail_idx, status.trade_agent_status_detail_idx)
        self.assertEqual(status2.cash_available, status2.cash_init - pos_status.margin - pos_status.commission)
        self.assertEqual(status2.position_value, pos_status.position_value)
        self.assertEqual(status2.curr_margin, pos_status.margin)
        self.assertEqual(status2.close_profit, 0)
        self.assertEqual(status2.position_profit, pos_status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, pos_status.floating_pl_cum)
        self.assertEqual(status2.commission_tot, pos_status.commission)
        self.assertEqual(status2.cash_init, status.cash_init)
        self.assertEqual(status2.cash_and_margin, status2.cash_available + status2.curr_margin)

    @staticmethod
    def add_trade_agent_status_detail():
        info = InitTest.add_stg_run_info()
        init_cash = 1000000
        md = {
            'ActionDay': '2018-12-14',
            'ActionTime': '13:24:35',
        }
        status = TradeAgentStatusDetail.create(info.stg_run_id, ExchangeName.DataIntegration, init_cash, md)
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(status)
            session.commit()
        return info, status

    @staticmethod
    def add_pos_status_detail(info):
        trade = PosStatusDetailTest.add_trade(info.stg_run_id)
        pos_status = PosStatusDetail.create_by_trade_detail(trade)
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(pos_status)
            session.commit()
        return pos_status


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
