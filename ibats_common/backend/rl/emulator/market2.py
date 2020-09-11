"""
@author  : MG
@Time    : 2019/6/28 17:09
@File    : market.py
@contact : mmmaaaggg@163.com
@desc    :
2020-01-29
在 market 基础上进行修改
调整 action 数值，为了更加适应 one_hot 模式，调整 long=0, short=1 close=2, keep=3
2020-09-05
调整属性名称
增加了 position_unit, long_holding_punish, punish_value, reward_multiplier 等参数化配置的地方
"""
from enum import IntEnum
import numpy as np
import pandas as pd

ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP = 0, 1, 2, 3
ACTIONS = [ACTION_LONG, ACTION_SHORT, ACTION_CLOSE, ACTION_KEEP]
ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP = 'long', 'short', 'close', 'keep'
ACTION_OPS = [ACTION_OP_LONG, ACTION_OP_SHORT, ACTION_OP_CLOSE, ACTION_OP_KEEP]
ACTION_OP_DIC = {_: ACTION_OPS[_] for _ in ACTIONS}
OP_ACTION_DIC = {ACTION_OPS[_]: _ for _ in ACTIONS}
# 内部访问的flag
_FLAG_LONG, _FLAG_SHORT, _FLAG_EMPTY = 1.0, -1.0, 0.0
# one-hot 模式的flag
FLAG_LONG, FLAG_SHORT, FLAG_EMPTY = _FLAG_LONG + 1, _FLAG_SHORT + 1, _FLAG_EMPTY + 1
FLAGS = [FLAG_LONG, FLAG_SHORT, FLAG_EMPTY]
DEFAULT_MD_OHLCVA_LABELS_MINIMAL = ['open', None, None, 'close', None, None]
DEFAULT_MD_OHLCVA_LABELS = ['open', 'high', 'low', 'close', 'volume', 'amount']
OPEN_INDEX, HIGH_INDEX, LOW_INDEX, CLOSE_INDEX, VOLUME_INDEX, AMOUNT_INDEX = 0, 1, 2, 3, 4, 5


class DealPriceScheme(IntEnum):
    """
    成交价形成机制
    """
    CURRENT_CLOSE = 0
    NEXT_OPEN = 1
    NEXT_OC_MIDDLE = 2  # 平均价格： (开盘价 + 收盘价) /2
    NEXT_BAR_CENTER = 3  # 重心价格： ((开盘价 + 收盘价) * 2 + 最高价 + 最低价) / 6


class QuotesMarket(object):
    def __init__(self, md_df: pd.DataFrame, data_factors, init_cash=2e5,
                 fee_rate=3e-3, position_unit=10, state_with_flag=False, reward_with_fee0=False,
                 md_ohlcva_labels=DEFAULT_MD_OHLCVA_LABELS_MINIMAL,
                 long_holding_punish=0, punish_value=0.01, reward_multiplier=1,
                 deal_price_scheme: DealPriceScheme = DealPriceScheme.NEXT_OPEN):
        """
        :param md_df: 行情数据
        :param data_factors: 因子数据,将作为 observation 返回
        :param init_cash: 初始现金
        :param fee_rate: 费率
        :param position_unit: 持仓单位
        :param state_with_flag: 默认False, observation 是否包含 多空标识,以及 当前 累计收益率 rr
        :param reward_with_fee0: 默认False, reward 是否包含 fee=0 状态的 reward
        :param md_ohlcva_labels: md 行情数据的 开、高、低、收、成交量、成交金额 标签，没有的可以为None，最少需要有 开盘 收盘 标签
        :param long_holding_punish: 默认为 0.0 数字是浮点型 np.float32
            数字大于0时，开始生效.持仓超过 限定数值后，开始对reward增加惩罚项。
            同时，返回的 observation 中包含连续持仓条数。
        :param punish_value: 默认为 0.01 惩罚值
        :param reward_multiplier: 默认为乘数 1.0
        :param deal_price_scheme: 成交价格方案，默认下一根K先开盘价
        :return:
        """
        # 参数有效性检查
        if deal_price_scheme == DealPriceScheme.NEXT_BAR_CENTER:
            assert md_ohlcva_labels[HIGH_INDEX] is not None
            assert md_ohlcva_labels[LOW_INDEX] is not None
        # 开高低收，vol，amount 序列
        self.data_close = md_df[md_ohlcva_labels[CLOSE_INDEX]]
        self.data_open = md_df[md_ohlcva_labels[OPEN_INDEX]]
        self.data_high = md_df[md_ohlcva_labels[HIGH_INDEX]] if md_ohlcva_labels[HIGH_INDEX] is not None else None
        self.data_low = md_df[md_ohlcva_labels[LOW_INDEX]] if md_ohlcva_labels[LOW_INDEX] is not None else None
        self.data_volume = md_df[md_ohlcva_labels[VOLUME_INDEX]] if md_ohlcva_labels[VOLUME_INDEX] is not None else None
        self.data_amount = md_df[md_ohlcva_labels[AMOUNT_INDEX]] if md_ohlcva_labels[AMOUNT_INDEX] is not None else None
        # 因子序列
        self.data_factor = data_factors
        self.action_operations = ACTION_OPS
        self.position_unit = position_unit
        self.fee_rate = fee_rate  # 千三手续费
        self.fee_curr_step = 0
        self.fee_tot = 0
        self.max_step_count = self.data_factor.shape[0] - 1
        self.init_cash = init_cash
        # reset use
        self.step_counter = 0
        self.cash = self.init_cash
        self.position_value = 0
        self.total_value = self.cash + self.position_value
        self.total_value_fee0 = self.cash + self.position_value
        self._flag = _FLAG_EMPTY
        self.state_with_flag = state_with_flag
        self.reward_with_fee0 = reward_with_fee0
        self.action_count = 0
        self._observation_latest = None
        self._done = False
        self._step_ret_latest = None
        self.long_holding_punish = np.float32(long_holding_punish)
        self._keep_holding_periods_len = 0.0
        self.punish_value = punish_value
        self.reward_multiplier = reward_multiplier
        self.deal_price_scheme = deal_price_scheme

    @property
    def flag(self):
        """外部访问 flag 标志位，为了方便 one_hot 模式，因此做 + 1 处理"""
        return self._flag + 1  # 为了方便转换成 one_hot 模式的 flag

    def reset(self):
        self.step_counter = 0
        self.cash = self.init_cash
        self.position_value = 0
        self.total_value = self.cash + self.position_value
        self.total_value_fee0 = self.cash + self.position_value
        self._flag = _FLAG_EMPTY
        self.fee_curr_step = 0
        self.fee_tot = 0
        self.action_count = 0
        self._observation_latest = self._get_observation_latest()
        self._done = False
        self._step_ret_latest = self._observation_latest, 0.0, self._done
        self._keep_holding_periods_len = 0.0
        return self._observation_latest

    def _get_observation_latest(self):
        """
        根据 step_counter state_with_flag, long_holding_punish 生成最新的 observation
        如果 仅返回 data_factor 的时候，则直接返回
        如果 带有 state_with_flag，或者 _keep_holding_periods_len 等状态信息是，则返回数组
        """
        observation_latest = [self.data_factor[self.step_counter]]
        if self.state_with_flag:
            rr = self.total_value / self.init_cash - 1
            # self._observation_latest = {'state': self.data_factor[self.step_counter],
            #                             'flag': np.array([self.flag]),
            #                             'rr': np.array([rr])}
            # 此处需要与 env.observation_spec 对应一致
            # 当前 env.observation_spec 不支持字典形式,因此只能使用数组形式
            observation_latest.append(np.array([self.flag], dtype=np.float32))
            observation_latest.append(np.array([rr], dtype=np.float32))

        if self.long_holding_punish > 0.0:
            observation_latest.append(np.array([self._keep_holding_periods_len], dtype=np.float32))

        if len(observation_latest) == 1:
            return observation_latest[0]
        else:
            return tuple(observation_latest)

    def observation_latest(self):
        return self._observation_latest

    def get_action_operations(self):
        return self.action_operations

    @property
    def get_deal_price(self):
        """
        获取当前成交价格
        """
        if self.deal_price_scheme == DealPriceScheme.NEXT_OPEN:
            price = self.data_open[self.step_counter]
        elif self.deal_price_scheme == DealPriceScheme.NEXT_OC_MIDDLE:
            price = (self.data_close[self.step_counter] + self.data_open[self.step_counter]) / 2
        elif self.deal_price_scheme == DealPriceScheme.NEXT_BAR_CENTER:
            price = ((self.data_close[self.step_counter] + self.data_open[self.step_counter]) * 2 +
                     self.data_high[self.step_counter] + self.data_low[self.step_counter]) / 6
        elif self.deal_price_scheme == DealPriceScheme.CURRENT_CLOSE:
            price = self.data_close[self.step_counter - 1]
        else:
            raise ValueError(f'当前价格成交机制 {self.deal_price_scheme} 不支持')

        return price

    def long(self):
        self._flag = _FLAG_LONG
        quotes = self.get_deal_price * self.position_unit
        self.cash -= quotes * (1 + self.fee_rate)
        self.position_value = quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1
        self._keep_holding_periods_len = 0.0

    def short(self):
        self._flag = _FLAG_SHORT
        quotes = self.get_deal_price * self.position_unit
        self.cash += quotes * (1 - self.fee_rate)
        self.position_value = - quotes
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1
        self._keep_holding_periods_len = 0.0

    def keep(self):
        quotes = self.get_deal_price * self.position_unit
        self.position_value = quotes * self._flag
        self._keep_holding_periods_len += 1.0

    def close_long(self):
        self._flag = _FLAG_EMPTY
        quotes = self.get_deal_price * self.position_unit
        self.cash += quotes * (1 - self.fee_rate)
        self.position_value = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1
        self._keep_holding_periods_len = 0.0

    def close_short(self):
        self._flag = _FLAG_EMPTY
        quotes = self.get_deal_price * self.position_unit
        self.cash -= quotes * (1 + self.fee_rate)
        self.position_value = 0
        self.fee_curr_step += quotes * self.fee_rate
        self.action_count += 1
        self._keep_holding_periods_len = 0.0

    @property
    def step_ret_latest(self):
        return self._step_ret_latest

    def step(self, action: int):
        if self._done:
            raise ValueError(f"It's Done state. max_step_count={self.max_step_count}, "
                             f"current step={self.step_counter}, total_value={self.total_value}")
        self.fee_curr_step = 0
        if action == ACTION_LONG:
            if self._flag == _FLAG_EMPTY:
                self.long()
            elif self._flag == _FLAG_SHORT:
                self.close_short()
                self.long()
            else:
                self.keep()

        elif action == ACTION_CLOSE:
            if self._flag == _FLAG_LONG:
                self.close_long()
            elif self._flag == _FLAG_SHORT:
                self.close_short()
            else:
                pass

        elif action == ACTION_SHORT:
            if self._flag == _FLAG_EMPTY:
                self.short()
            elif self._flag == _FLAG_LONG:
                self.close_long()
                self.short()
            else:
                self.keep()
        elif action == ACTION_KEEP:
            self.keep()
        else:
            raise ValueError(f"action={action} should be one of keys {ACTIONS}")

        # 计算费用
        self.fee_tot += self.fee_curr_step
        self.total_value_fee0 = self.total_value + self.fee_tot

        # 计算价值
        price = self.data_close[self.step_counter]
        position_value = price * self.position_unit * self._flag
        # self.total_value 此时记录的是上一轮操作结果时的总资产.
        # 当前现金流+持仓价值 - 上一轮的总资产 = 当期资产净增长
        net_reward = self.cash + position_value - self.total_value
        self.step_counter += 1
        self.total_value = position_value + self.cash

        if self.total_value < price:
            self._done = True
        if self.step_counter >= self.max_step_count:
            self._done = True

        self._observation_latest = self._get_observation_latest()
        if self.reward_with_fee0:
            # 计算含fee reward
            _reward_latest1 = net_reward / price / self.position_unit
            _reward_latest2 = (net_reward + self.fee_curr_step) / price / self.position_unit

            if 0.0 < self.long_holding_punish < self._keep_holding_periods_len:
                # 持仓周期超过 long_holding_punish，对 reward 增加 惩罚项
                reward_latest = - self.punish_value, - self.punish_value
            else:
                reward_latest = _reward_latest1, _reward_latest2

        else:
            # 仅计算计费的reward
            if 0.0 < self.long_holding_punish < self._keep_holding_periods_len:
                # 持仓超过周期限制后 reward 返回固定惩罚值
                reward_latest = -self.punish_value
            else:
                reward_latest = net_reward / price / self.position_unit

        reward_latest *= self.reward_multiplier
        self._step_ret_latest = self._observation_latest, reward_latest, self._done
        return self._step_ret_latest


def _test_quote_market():
    import os
    n_step = 60
    from ibats_common.example.data import load_data
    md_df = load_data(
        'RB.csv',
        folder_path=os.path.join(os.pardir, os.pardir, os.pardir, 'example', 'data')  # r'..\..\..\example\data'
    ).set_index('trade_date')[DEFAULT_MD_OHLCVA_LABELS]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]
    # 建立 QuotesMarket
    qm = QuotesMarket(md_df=md_df[['close', 'open']], data_factors=data_arr_batch, state_with_flag=True)
    next_observation = qm.reset()
    assert len(next_observation) == 3
    assert next_observation[0].shape[0] == n_step
    assert next_observation[1] == FLAG_EMPTY
    next_observation, reward, done = qm.step(ACTION_LONG)
    assert len(next_observation) == 3
    assert next_observation[1] == FLAG_LONG
    assert not done
    next_observation, reward, done = qm.step(ACTION_CLOSE)
    assert next_observation[1] == FLAG_EMPTY
    assert reward != 0
    next_observation, reward, done = qm.step(ACTION_CLOSE)
    assert next_observation[1] == FLAG_EMPTY
    assert reward == 0
    next_observation, reward, done = qm.step(ACTION_KEEP)
    assert next_observation[1] == FLAG_EMPTY
    assert reward == 0
    next_observation, reward, done = qm.step(ACTION_SHORT)
    assert next_observation[1] == FLAG_SHORT
    assert not done
    next_observation, reward, done = qm.step(ACTION_KEEP)
    assert next_observation[1] == FLAG_SHORT
    assert reward != 0
    try:
        qm.step(4)
    except ValueError:
        print('is ok for not supporting action>3')


if __name__ == "__main__":
    _test_quote_market()
