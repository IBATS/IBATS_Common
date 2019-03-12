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
from datetime import date, datetime, time
import unittest

from ibats_common.common import ExchangeName


class InitTest(unittest.TestCase):  # 继承unittest.TestCase
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
        info = InitTest.add_stg_run_info()

        self.assertIsNotNone(info.stg_run_id)
        self.assertGreater(info.stg_run_id, -1)

        with with_db_session(engine_ibats) as session:
            info2 = session.query(StgRunInfo).filter(StgRunInfo.stg_run_id == info.stg_run_id).first()

        self.assertIsInstance(info2, StgRunInfo)
        self.assertEqual(info.stg_run_id, info2.stg_run_id)

    @staticmethod
    def add_order(stg_run_id=None):
        if stg_run_id is None:
            info = InitTest.add_stg_run_info()
            stg_run_id = info.stg_run_id

        order = OrderDetail(stg_run_id, trade_agent_key=ExchangeName.DataIntegration,
                            order_dt=datetime.now(), order_date=date.today(), order_time=datetime.now().time(),
                            order_millisec=99, direction=int(Direction.Long), action=int(Action.Open), symbol='RB1801',
                            order_price=1000.0, order_vol=20
                            )
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(order)
            session.commit()
        return order

    def test_add_order_detail(self):
        order = InitTest.add_order()
        self.assertIsNotNone(order)
        self.assertGreater(order.order_idx, -1)

        with with_db_session(engine_ibats) as session:
            order2 = session.query(OrderDetail).filter(
                OrderDetail.order_idx == order.order_idx,
                OrderDetail.stg_run_id == order.stg_run_id
            ).first()

        self.assertIsInstance(order2, OrderDetail)
        self.assertEqual(order2.order_idx, order.order_idx)

    def test_add_trade_detail(self):
        trade = InitTest.add_trade()

        self.assertIsNotNone(trade)
        self.assertGreater(trade.trade_idx, -1)

        with with_db_session(engine_ibats) as session:
            trade2 = session.query(TradeDetail).filter(
                TradeDetail.trade_idx == trade.trade_idx,
                TradeDetail.stg_run_id == trade.stg_run_id
            ).first()

        self.assertIsInstance(trade2, TradeDetail)
        self.assertEqual(trade.trade_idx, trade2.trade_idx)

    @staticmethod
    def add_trade():
        order = InitTest.add_order()
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
        pos_status = InitTest.add_pos_status()

        self.assertIsNotNone(pos_status)
        self.assertGreater(pos_status.pos_status_detail_idx, -1)

        with with_db_session(engine_ibats) as session:
            pos_status2 = session.query(PosStatusDetail).filter(
                PosStatusDetail.pos_status_detail_idx == pos_status.pos_status_detail_idx,
                PosStatusDetail.stg_run_id == pos_status.stg_run_id
            ).first()

        self.assertIsInstance(pos_status2, PosStatusDetail)
        self.assertEqual(pos_status.pos_status_detail_idx, pos_status2.pos_status_detail_idx)

    @staticmethod
    def add_pos_status():
        trade = InitTest.add_trade()
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
        status = InitTest.add_trade_agent_status()

        self.assertIsNotNone(status)
        self.assertGreater(status.trade_agent_status_detail_idx, -1)

        with with_db_session(engine_ibats) as session:
            status2 = session.query(TradeAgentStatusDetail).filter(
                TradeAgentStatusDetail.trade_agent_status_detail_idx == status.trade_agent_status_detail_idx,
                TradeAgentStatusDetail.stg_run_id == status.stg_run_id
            ).first()

        self.assertIsInstance(status2, TradeAgentStatusDetail)
        self.assertEqual(status.trade_agent_status_detail_idx, status2.trade_agent_status_detail_idx)

    @staticmethod
    def add_trade_agent_status():
        pos_status = InitTest.add_pos_status()
        cash_init = 100000
        cash_available = cash_init - pos_status.margin - pos_status.commission
        status = TradeAgentStatusDetail(
            pos_status.stg_run_id, trade_agent_key=pos_status.trade_agent_key,
            trade_dt=pos_status.trade_dt, trade_date=pos_status.trade_date,
            trade_time=pos_status.trade_time,
            trade_millisec=pos_status.trade_millisec,
            cash_available=cash_available,
            curr_margin=pos_status.margin,
            close_profit=0, position_profit=pos_status.floating_pl,
            floating_pl_cum=pos_status.floating_pl_cum,
            commission_tot=pos_status.commission_tot,
            cash_init=cash_init
            )
        status.cash_and_margin_value = cash_available + pos_status.margin
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(status)
            session.commit()
        return status

    def test_add_stg_run_detail(self):
        detail = InitTest.add_stg_run_detail()

        self.assertIsNotNone(detail)
        self.assertGreater(detail.stg_run_status_detail_idx, -1)

        with with_db_session(engine_ibats) as session:
            detail2 = session.query(StgRunStatusDetail).filter(
                StgRunStatusDetail.stg_run_status_detail_idx == detail.stg_run_status_detail_idx,
                StgRunStatusDetail.stg_run_id == detail.stg_run_id
            ).first()

        self.assertIsInstance(detail2, StgRunStatusDetail)
        self.assertEqual(detail.stg_run_status_detail_idx, detail2.stg_run_status_detail_idx)

    @staticmethod
    def add_stg_run_detail():
        status = InitTest.add_trade_agent_status()
        detail = StgRunStatusDetail(
            status.stg_run_id, trade_dt=status.trade_dt, trade_date=status.trade_date,
            trade_time=status.trade_time,
            trade_millisec=status.trade_millisec, cash_available=0,
            curr_margin=status.curr_margin,
            close_profit=status.close_profit, position_profit=status.position_profit,
            floating_pl_cum=status.floating_pl_cum,
            commission_tot=status.commission_tot, cash_init=status.cash_init,
            balance_tot=status.cash_and_margin)
        with with_db_session(engine_ibats, expire_on_commit=False) as session:
            session.add(detail)
            session.commit()
        return detail


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
        from ibats_common.config import update_db_config, ConfigBase
        update_db_config({
            ConfigBase.DB_SCHEMA_IBATS: 'mysql://mg:Dcba1234@localhost/' + ConfigBase.DB_SCHEMA_IBATS,
        })
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
        from ibats_common.config import update_db_config, ConfigBase
        update_db_config({
            ConfigBase.DB_SCHEMA_IBATS: 'mysql://mg:Dcba1234@localhost/' + ConfigBase.DB_SCHEMA_IBATS,
        })
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
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status2.margin * 10000 // 1)  # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
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
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status2.margin * 10000 // 1)  # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
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
        self.assertEqual(status2.floating_pl_rate * 10000 // 1, -(
                    trade.commission + trade2.commission) / status.margin * 10000 // 1)  # 初次建仓时浮动收益就等于手续费 / 保证金 "*10000//1" 指在 万分之一以上精度相等即可
        self.assertEqual(status2.floating_pl_chg, status2.floating_pl - status.floating_pl)
        self.assertEqual(status2.floating_pl_cum, status2.floating_pl)
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
        from ibats_common.config import update_db_config, ConfigBase
        update_db_config({
            ConfigBase.DB_SCHEMA_IBATS: 'mysql://mg:Dcba1234@localhost/' + ConfigBase.DB_SCHEMA_IBATS,
        })
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        init()
        global engine_ibats
        engine_ibats = engines.engine_ibats

    def test_create_by_trade_agent_status_detail_list(self):
        detail = TradeAgentStatusDetailTest.add_trade_agent_status_detail()



if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
