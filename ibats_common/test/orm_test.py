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
from datetime import date, datetime
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
                            order_dt=datetime.now(), order_date=date.today(), order_time=datetime.now().time(),
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
        self.assertEqual(trade.trade_idx, trade2.trade_idx)

    @staticmethod
    def add_trade():
        order = MyTest.add_order()
        trade = TradeDetail(order.stg_run_id, trade_agent_key=ExchangeName.DataIntegration, order_idx=order.order_idx,
                            order_price=1000.0, order_vol=20,
                            trade_dt=datetime.now(), trade_date=date.today(), trade_time=datetime.now().time(),
                            trade_millisec=99, direction=int(Direction.Long), action=int(Action.Open), symbol='RB1801',
                            trade_price=order.order_price + 1, trade_vol=order.order_vol,
                            margin=order.order_price * order.order_vol,
                            commission=order.order_price * order.order_vol + 0.00005, multiple=10, margin_ratio=1
                            )
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(trade)
            session.commit()
        return trade

    def test_add_pos_status(self):
        pos_status = MyTest.add_pos_status()

        self.assertIsNotNone(pos_status)
        self.assertGreater(pos_status.pos_status_detail_idx, -1)

        with with_db_session(engine_ibats) as session:
            pos_status2 = session.query(PosStatusDetail).filter(
                PosStatusDetail.pos_status_detail_idx == pos_status.pos_status_detail_idx).first()

        self.assertIsInstance(pos_status2, PosStatusDetail)
        self.assertEqual(pos_status.pos_status_detail_idx, pos_status2.pos_status_detail_idx)

    @staticmethod
    def add_pos_status():
        trade = MyTest.add_trade()
        commission_rate = 0.0005
        multiple, margin_ratio = 10, 0.12
        pos_status = PosStatusDetail(
            trade.stg_run_id, trade_agent_key=trade.trade_agent_key, trade_idx=trade.trade_idx,
            trade_dt=trade.trade_dt, trade_date=trade.trade_date, trade_time=trade.trade_time,
            trade_millisec=trade.trade_millisec, direction=trade.direction,
            symbol=trade.symbol,
            position=trade.trade_vol, position_chg=-trade.trade_vol,
            avg_price=trade.trade_price,
            cur_price=trade.trade_price, floating_pl=-1, floating_pl_rate=-commission_rate,
            floating_pl_chg=trade.trade_price * trade.trade_vol * trade.multiple * commission_rate,
            floating_pl_cum=-13, rr=-commission_rate,
            margin=trade.trade_price * trade.trade_vol,
            margin_chg=-trade.trade_price * trade.trade_vol,
            position_date_type=PositionDateType.Today.value,
            commission=trade.trade_vol * trade.trade_price * commission_rate * trade.multiple,
            commission_tot=trade.trade_vol * trade.trade_price * commission_rate * trade.multiple,
            multiple=multiple,
            margin_ratio=margin_ratio,
        )
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(pos_status)
            session.commit()
        return pos_status

    def test_add_trade_agent_status(self):
        status = MyTest.add_trade_agent_status()

        self.assertIsNotNone(status)
        self.assertGreater(status.trade_agent_status_detail_idx, -1)

        with with_db_session(engine_ibats) as session:
            status2 = session.query(TradeAgentStatusDetail).filter(
                TradeAgentStatusDetail.trade_agent_status_detail_idx == status.trade_agent_status_detail_idx).first()

        self.assertIsInstance(status2, TradeAgentStatusDetail)
        self.assertEqual(status.trade_agent_status_detail_idx, status2.trade_agent_status_detail_idx)

    @staticmethod
    def add_trade_agent_status():
        pos_status = MyTest.add_pos_status()
        status = TradeAgentStatusDetail(
            pos_status.stg_run_id, trade_agent_key=pos_status.trade_agent_key,
            trade_dt=pos_status.trade_dt, trade_date=pos_status.trade_date,
            trade_time=pos_status.trade_time,
            trade_millisec=pos_status.trade_millisec, available_cash=0,
            curr_margin=pos_status.margin,
            close_profit=0, position_profit=pos_status.floating_pl,
            floating_pl_cum=pos_status.floating_pl_cum,
            commission_tot=pos_status.commission_tot, balance_init=10000,
            balance_tot=pos_status.cur_price * pos_status.position * pos_status.multiple)
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(status)
            session.commit()
        return status


if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
