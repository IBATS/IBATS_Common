#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-4-19 上午8:40
@File    : ai_stg.py
@contact : mmmaaaggg@163.com
@desc    : 简单的 RNN LSTM 构建策略模型，运行该模型需要首先安装 TensorFlow 包
pip3 install tensorflow sklearn tflearn
"""
import os
from ibats_utils.mess import get_last_idx, get_folder_path
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
import numpy as np
import random

from ibats_common import module_root_path
from ibats_common.analysis.plot_db import show_rr_with_md
from ibats_common.common import PeriodType, RunMode, BacktestTradeMode, ExchangeName, ContextKey, Direction, CalcMode
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.td_agent import *
from ibats_local_trader.agent.md_agent import *

logger = logging.getLogger(__name__)


class AIStg(StgBase):

    def __init__(self, unit=1, train=True):
        super().__init__()
        self.unit = unit
        self.input_size = 13
        self.batch_size = 50
        self.n_step = 20
        self.output_size = 2
        self.n_hidden_units = 10
        self.lr = 0.006
        self.normalization_model = True
        self._model = None
        # tf.Session()
        self._session = None
        self.train_validation_rate = 0.8
        self.is_load_model_if_exist = False
        self.training_iters = 600
        self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = None, None, None, None
        self.classify_wave_rate = 0.0033
        self.predict_test_random_state = 1
        folder_path = get_folder_path('my_net', create_if_not_found=False)
        file_path = os.path.join(folder_path, f"net_wr_{int(self.classify_wave_rate*10000)}.tfl")
        self.model_file_path = file_path

    @property
    def session(self):
        if self._session is None:
            if self.model is None:
                raise ValueError('model 需要先于 session 被创建')
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session

    @property
    def model(self) -> tflearn.models.DNN:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def get_factors(self, md_df: pd.DataFrame, tail_n=None):
        if tail_n is not None:
            md_df = md_df.tail(tail_n)
        df = md_df[~md_df['close'].isnull()][[
            'open', 'high', 'low', 'close', 'volume', 'oi', 'warehousewarrant', 'termstructure']]
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['pct_change_vol'] = df['volume'].pct_change()
        df['pct_change'] = df['close'].pct_change()

        factors = df.fillna(0).to_numpy()
        if self.input_size is None or self.input_size != factors.shape[1]:
            self.input_size = factors.shape[1]
            self.n_hidden_units = self.input_size * 2
            logger.info("set input_size: %d", self.input_size)
            logger.info("set n_hidden_units: %d", self.n_hidden_units)

        # if self.normalization_model:
        #     factors = (factors - np.mean(factors, 0)) / np.std(factors, 0)

        return factors

    def get_x_y(self, md_df):
        factors = self.get_factors(md_df)
        price_arr = factors[:, 0]
        self.input_size = factors.shape[1]
        # ys_all = self.calc_y_against_future_data(price_arr, -0.01, 0.01)
        ys_all = self.calc_y_against_future_data(price_arr, -self.classify_wave_rate, self.classify_wave_rate)
        idx_last_available_label = get_last_idx(ys_all, lambda x: x.sum() == 0)
        factors = factors[:idx_last_available_label + 1, :]
        range_from = self.n_step - 1
        range_to = idx_last_available_label + 1
        xs = np.zeros((range_to - range_from, self.n_step, self.input_size))
        for num, index in enumerate(range(range_from, range_to)):
            xs[num, :, :] = factors[(index - self.n_step + 1):(index + 1), :]

        ys = ys_all[range_from:range_to, :]

        return xs, ys

    def calc_y_against_future_data(self, value_arr: np.ndarray, min_pct: float, max_pct: float, max_future=None):
        """
        根据时间序列数据 pct_arr 计算每一个时点目标标示 -1 0 1
        计算方式：
        当某一点未来波动首先 > 上届 min_pct，则标记为： [0, 1]
        当某一点未来波动首先 < 下届 max_pct，则标记为： [1, 0]
        :param value_arr:
        :param min_pct:
        :param max_pct:
        :param max_future:最大搜索长度
        :param output_size:最大搜索长度
        :return:
        """

        value_arr[np.isnan(value_arr)] = 0
        arr_len = value_arr.shape[0]
        target_arr = np.zeros((arr_len, self.output_size))
        for i in range(arr_len):
            base = value_arr[i]
            for j in range(i + 1, arr_len):
                result = value_arr[j] / base - 1
                if result < min_pct:
                    target_arr[i, 0] = 1
                    break
                elif result > max_pct:
                    target_arr[i, 1] = 1
                    break
        return target_arr

    def get_batch_xs(self, factors: np.ndarray, index=None):
        """
        取 batch_xs
        :param factors:
        :param index: 样本起始坐标，如果为None，则默认取尾部一组样本
        :return:
        """
        if index is None:
            index = factors.shape[0] - 1

        batch_xs = np.zeros((1, self.n_step, self.input_size))
        batch_xs[0, :, :] = factors[(index - self.n_step + 1):(index + 1), :]

        return batch_xs

    def get_batch_by_random(self, factors: np.ndarray, labels: np.ndarray):
        """
        够在一系列输入输出数据集
        xs： 两条同频率，不同位移的sin曲线
        ys_value： 目标是一条cos曲线
        ys: ys_value 未来涨跌标识
        i_s：X 序列
        """
        xs = np.zeros((self.batch_size, self.n_step, self.input_size))
        ys = np.zeros((self.batch_size, self.output_size))
        # available_batch_size, num = 0, 0
        samples_index = random.sample(range(self.n_step - 1, factors.shape[0] - 1), self.batch_size)
        examples_index_list = []
        for available_batch_size, index in enumerate(samples_index):
            tmp = factors[(index - self.n_step + 1):(index + 1), :]
            if tmp.shape[0] < self.n_step:
                break
            xs[available_batch_size, :, :] = tmp
            ys[available_batch_size, :] = labels[index, :]
            examples_index_list.append(index)
            if available_batch_size + 1 >= self.batch_size:
                available_batch_size += 1
                break

        # returned xs, ys_value and shape (batch, step, input)
        return xs, ys, examples_index_list

    def _build_model(self) -> tflearn.models.DNN:
        # hyperparameters
        # lr = LR
        # batch_size = BATCH_SIZE
        #
        # n_inputs = INPUT_SIZE  # MNIST data input (img shape 28*28)
        # n_step = TIME_STEPS  # time steps
        # n_hidden_units = CELL_SIZE  # neurons in hidden layer
        # n_classes = OUTPUT_SIZE  # MNIST classes (0-9 digits)
        # Network building
        net = tflearn.input_data([None, self.n_step, self.input_size])
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.lstm(net, self.n_hidden_units, dropout=0.8)
        net = tflearn.fully_connected(net, self.output_size, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

        # Training
        tensorboard_dir = os.path.join(module_root_path, 'tflearn_logs')
        _model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path='model.tfl.ckpt',
                             tensorboard_dir=tensorboard_dir)
        return _model

    def train(self, md_df):
        xs, ys = self.get_x_y(md_df)
        # xs_train, xs_validation, ys_train, ys_validation = self.separate_train_validation(xs, ys)
        xs_train, xs_validation, ys_train, ys_validation = train_test_split(
            xs, ys, test_size=0.2, random_state=self.predict_test_random_state)
        self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = xs_train, xs_validation, ys_train, ys_validation
        # model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)
        sess = self.session
        with sess.as_default():
            tflearn.is_training(True)
            self.model.fit(xs_train, ys_train, validation_set=(xs_validation, ys_validation),
                           show_metric=True, batch_size=32, n_epoch=4)
            tflearn.is_training(False)
        return self.model

    def save_model(self):
        """
        将模型导出到文件
        :return:
        """
        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        # save_path = saver.save(self.session, self.model_file_path)
        self.model.save(self.model_file_path)
        logger.info("模型保存到: %s", self.model_file_path)
        return self.model_file_path

    def load_model_if_exist(self):
        """
        将模型导出到文件
        :return:
        """
        if self.is_load_model_if_exist and self.model_file_exists():
            # 检查文件是否存在
            model = self.model      # 这句话是必须的，需要实现建立模型才可以加载
            # sess = self.session
            # saver = tf.train.Saver(tf.trainable_variables())
            # save_path = saver.restore(sess, self.model_file_path)
            model.load(self.model_file_path)
            logger.info("load from path: %s", self.model_file_path)
            return True

        return False

    def predict_test(self, md_df):
        logger.info('开始预测')
        # sess = self.session
        if self.xs_train is None:
            xs, ys = self.get_x_y(md_df)
            xs_train, xs_validation, ys_train, ys_validation = train_test_split(xs, ys, test_size=0.2, random_state=1)
            self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = xs_train, xs_validation, ys_train, ys_validation

        logger.info("批量预测")
        result = self.model.evaluate(self.xs_validation, self.ys_validation, batch_size=self.batch_size)
        logger.info("accuracy: %.2f%%" % (result[0] * 100))

        logger.info("批量预测2")
        real_ys = np.argmax(self.ys_validation, axis=1)
        # pred_ys = np.argmax(self.model.predict_label(self.xs_validation), axis=1) 与 evaluate 结果刚好相反
        # 因此不使用 predict_label 函数
        pred_ys = np.argmax(self.model.predict(self.xs_validation), axis=1)
        logger.info("accuracy: %.2f%%" % (sum(pred_ys == real_ys) / len(pred_ys) * 100))
        logger.info("pred/real: \n%s\n%s", pred_ys, real_ys)

        logger.info("独立样本预测(predict_latest)")
        pred_ys = []
        for idx, y in enumerate(self.ys_validation):
            x = self.xs_validation[idx:idx+1, :, :]
            pred_y = self.model.predict(x)
            pred_ys.extend(np.argmax(pred_y, axis=1))

        pred_ys = np.array(pred_ys)
        logger.info("accuracy: %.2f%%" % (sum(pred_ys == real_ys) / len(pred_ys) * 100))
        logger.info("pred: \n%s\n%s", pred_ys, real_ys)

    def predict_latest(self, md_df):
        """
        计算最新一个 X，返回分类结果
        二分类，返回 0 / 1
        三分类，返回 0 / 1 / 2
        :param md_df:
        :return:
        """
        factors = self.get_factors(md_df, tail_n=self.n_step)
        x = self.get_batch_xs(factors)
        pred_y = np.argmax(self.model.predict(x), axis=1)[-1]
        # is_buy, is_sell = pred_mark == 1, pred_mark == 0
        # return is_buy, is_sell
        return pred_y

    def on_prepare_min1(self, md_df, context):
        if md_df is None:
            return

        # 加载模型
        is_load = self.load_model_if_exist()

        if not is_load:
            # 训练模型
            self.train(md_df)
            self.save_model()

        self.predict_test(md_df)

    def on_min1(self, md_df, context):
        pred_mark = self.predict_latest(md_df)
        is_buy, is_sell = pred_mark == 1, pred_mark == 0
        # trade_date = md_df['trade_date'].iloc[-1]
        # logger.info('%s is_buy=%s, is_sell=%s', trade_date, str(is_buy), str(is_sell))
        close = md_df['close'].iloc[-1]
        instrument_id = context[ContextKey.instrument_id_list][0]
        if is_buy:  # is_buy
            position_date_pos_info_dic = self.get_position(instrument_id)
            no_target_position = True
            if position_date_pos_info_dic is not None:
                for position_date, pos_info in position_date_pos_info_dic.items():
                    direction = pos_info.direction
                    if direction == Direction.Short:
                        self.close_short(instrument_id, close, pos_info.position)
                    elif direction == Direction.Long:
                        no_target_position = False
            if no_target_position:
                self.open_long(instrument_id, close, self.unit)
            else:
                logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

        if is_sell:  # is_sell
            position_date_pos_info_dic = self.get_position(instrument_id)
            no_holding_target_position = True
            if position_date_pos_info_dic is not None:
                for position_date, pos_info in position_date_pos_info_dic.items():
                    direction = pos_info.direction
                    if direction == Direction.Long:
                        self.close_long(instrument_id, close, pos_info.position)
                    elif direction == Direction.Short:
                        no_holding_target_position = False
            if no_holding_target_position:
                self.open_short(instrument_id, close, self.unit)
            else:
                logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

    def model_file_exists(self):
        folder_path, file_name = os.path.split(self.model_file_path)
        if not os.path.exists(folder_path):
            return False
        for f_name in os.listdir(folder_path):
            if f_name.find(file_name) == 0:
                return True

        return False


def _test_use(is_plot):
    from ibats_common import module_root_path
    import os
    # 参数设置
    run_mode = RunMode.Backtest
    strategy_params = {'unit': 1}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': ['RB'],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2013-1-1',
        # 'C:\GitHub\IBATS_Common\ibats_common\example\ru_price2.csv'
        'file_path': os.path.abspath(os.path.join(module_root_path, 'example', 'data', 'RB.csv')),
        'symbol_key': 'instrument_type',
    }]
    if run_mode == RunMode.Realtime:
        trade_agent_params = {
        }
        strategy_handler_param = {
        }
    else:
        trade_agent_params = {
            'trade_mode': BacktestTradeMode.Order_2_Deal,
            'init_cash': 10000,
            "calc_mode": CalcMode.Margin,
        }
        strategy_handler_param = {
            'date_from': '2013-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
            'date_to': '2018-10-18',
        }
    # 初始化策略处理器
    stghandler = strategy_handler_factory(
        stg_class=AIStg,
        strategy_params=strategy_params,
        md_agent_params_list=md_agent_params_list,
        exchange_name=ExchangeName.LocalFile,
        run_mode=RunMode.Backtest,
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
        from ibats_common.analysis.plot_db import show_order, show_cash_and_margin
        show_order(stg_run_id)
        show_cash_and_margin(stg_run_id)
        show_rr_with_md(stg_run_id)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    is_plot = True
    _test_use(is_plot)
