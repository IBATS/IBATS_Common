#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-4-28 下午5:30
@File    : ai_3_stg.py
@contact : mmmaaaggg@163.com
@desc    : 简单的 RNN LSTM 构建策略模型，运行该模型需要首先安装 TensorFlow 包
pip3 install tensorflow
在 ai_stg.py 基础上进行三分类训练
"""

import os
from ibats_utils.mess import get_last_idx, get_folder_path
import tensorflow as tf
import numpy as np
import random
from ibats_common.common import PeriodType, RunMode, BacktestTradeMode, ExchangeName, ContextKey, Direction, CalcMode
from ibats_common.strategy import StgBase
from ibats_common.strategy_handler import strategy_handler_factory
from ibats_local_trader.agent.td_agent import *
from ibats_local_trader.agent.md_agent import *

logger = logging.getLogger(__name__)


class LSTMRNN:
    def __init__(self, n_step, n_inputs, n_hidden_units, n_classes, lr, batch_size, normalization_model):
        """

        :param n_step: time steps
        :param n_inputs: MNIST data input (img shape 28*28)
        :param n_hidden_units: neurons in hidden layer
        :param n_classes: MNIST classes (0-9 digits)
        :param lr:
        :param training_iters:
        :param batch_size:
        :param normalization_model:
        """
        self.n_step = n_step
        self.n_inputs = n_inputs
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.normalization_model = normalization_model
        # attributes defined in other functions
        self.cost = None
        self.l_in_y = None
        self.cell_outputs = None
        self.cell_init_state = None
        self.cell_final_state = None

        with tf.name_scope('inputs'):
            # tf Graph input
            self.xs = tf.placeholder(tf.float32, [None, n_step, n_inputs])
            self.ys = tf.placeholder(tf.float32, [None, n_classes])
            self.is_training = tf.placeholder(tf.bool, [])

        # Define weights
        self.weights = {
            # 28*128
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }

        self.biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.ys, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def add_input_layer(self):
        # hidden layer for input to cell
        # X (128 batch, 28 steps, 28 inputs)
        # ==> X (128 * 28, 28 inputs)
        self.l_in_x = tf.reshape(self.xs, [-1, self.n_inputs])
        self.Ws_in = tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units]))
        self.bs_in = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ]))
        # ==> X_in (128 batch * 28 steps, 128 hidden)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(self.l_in_x, self.Ws_in) + self.bs_in
            if self.normalization_model:
                l_in_y = tf.layers.batch_normalization(l_in_y, training=self.is_training)

        # ==> X_in (128 batch, 28 steps, 128 hidden)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_step, self.n_hidden_units])

    def add_cell(self):
        # cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, time_major=False, dtype=tf.float32)

    def add_output_layer(self):
        # hidden layer for output as the final results
        # method 1
        # results = tf.matmul(states[1], weights['out']) + biases['out']  # states[1] is m_state
        # method 2
        # unpack to list[(batch, outputs)...] * steps
        self.l_out_x = tf.unstack(tf.transpose(self.cell_outputs, [1, 0, 2]))  # states is the last outputs
        self.Ws_out = tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        self.bs_out = tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        self.pred = tf.matmul(self.l_out_x[-1], self.Ws_out) + self.bs_out

    def compute_cost(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pred, labels=self.ys))
        tf.summary.scalar('cost', self.cost)


class AIStg(StgBase):

    def __init__(self, unit=1, train=True):
        super().__init__()
        self.unit = unit
        self.is_load_model_if_exist = True
        self.input_size = 4
        self.batch_size = 50
        self.n_step = 60
        self.output_size = 3
        self.n_hidden_units = 8
        self.lr = 0.006
        self.normalization_model = True
        self._model = None
        self.model_file_path = None
        # tf.Session()
        self._session = None
        self.train_validation_rate = 0.8

    @property
    def session(self):
        if self._session is None:
            if self.model is None:
                raise ValueError('model 需要先于 session 被创建')
            self._session = tf.Session()
        return self._session

    @property
    def model(self):
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def get_factors(self, md_df: pd.DataFrame, tail_n=None) -> np.ndarray:
        if tail_n is not None:
            md_df = md_df.tail(tail_n)
        df = md_df[~md_df['close'].isnull()][[
            'open', 'high', 'low', 'close', 'volume', 'oi', 'warehousewarrant', 'termstructure']]

        factors = df.fillna(0).to_numpy()
        if self.input_size is None or self.input_size != factors.shape[1]:
            self.input_size = factors.shape[1]
            self.logger.info("set input_size: %d", self.input_size)

        if self.normalization_model:
            factors = (factors - np.mean(factors, 0)) / np.std(factors, 0)

        return factors

    def get_factors_with_labels(self, md_df):
        factors = self.get_factors(md_df)
        price_arr = factors[:, 0]
        self.input_size = factors.shape[1]
        max_future = 5
        labels = self.calc_label_with_future_value(price_arr, -0.01, 0.01, max_future=max_future)
        # idx_last_available_label = get_last_idx(labels, lambda x: x.sum() == 0)
        factors = factors[:-max_future, :]
        labels = labels[:-max_future, :]

        return factors, labels

    def calc_label_with_future_value(self, value_arr: np.ndarray, min_pct: float, max_pct: float, max_future=None):
        """
        根据时间序列数据 pct_arr 计算每一个时点目标标示 -1 0 1
        计算方式：
        当某一点未来波动首先 > 上届 min_pct，则标记为：   [0, 0, 1]
        超过最大检测日期，则被认为是震荡势，则标记为:      [0, 1, 0]
        当某一点未来波动首先 < 下届 max_pct，则标记为：   [1, 0, 0]
        :param value_arr:
        :param min_pct:
        :param max_pct:
        :param max_future:最大搜索长度
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
                    target_arr[i, 2] = 1
                    break
                elif max_future is not None and j-i >= max_future:
                    # 超过最大检测日期，则被认为是震荡势
                    target_arr[i, 1] = 1
                    break
        return target_arr

    def get_batch_xs(self, factors: np.ndarray, num=None):
        """
        取 batch_xs
        :param factors:
        :param num: 样本起始坐标，如果为None，则默认取尾部一组样本
        :return:
        """
        if num is None:
            num = factors.shape[0] - self.n_step

        batch_xs = np.zeros((1, self.n_step, self.input_size))
        batch_xs[0, :, :] = factors[num:num + self.n_step, :]

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
        samples = random.sample(range(self.n_step, factors.shape[0]), self.batch_size)
        available_examples = []
        for available_batch_size, num in enumerate(samples):
            tmp = factors[(num - self.n_step):num, :]
            if tmp.shape[0] < self.n_step:
                break
            xs[available_batch_size, :, :] = tmp
            ys[available_batch_size, :] = labels[num, :]
            available_examples.append(num)
            if available_batch_size + 1 >= self.batch_size:
                available_batch_size += 1
                break

        # returned xs, ys_value and shape (batch, step, input)
        return xs, ys, available_examples

    def build_model(self):
        # hyperparameters
        # lr = LR
        # batch_size = BATCH_SIZE
        #
        # n_inputs = INPUT_SIZE  # MNIST data input (img shape 28*28)
        # n_step = TIME_STEPS  # time steps
        # n_hidden_units = CELL_SIZE  # neurons in hidden layer
        # n_classes = OUTPUT_SIZE  # MNIST classes (0-9 digits)
        model = LSTMRNN(
            self.n_step, self.input_size, self.n_hidden_units, self.output_size, self.lr, self.batch_size,
            self.normalization_model)
        return model

    def train(self, md_df):
        factors, labels = self.get_factors_with_labels(md_df)
        factors_train, factors_validation, labels_train, labels_validation = self.separate_train_validation(
            factors, labels)
        training_iters = 500

        sess = self.session
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', sess.graph)
        # relocate to the local dir and run this line to view it on Chrome(http://0.0.0.0:6006/):
        # $ tensorboard --logdir='logs'

        sess.run(tf.global_variables_initializer())
        step = 0
        while step < training_iters:  # * self.model.batch_size
            # batch_xs, batch_ys = mnist.train.next_batch(model.batch_size)
            # batch_xs.shape, batch_ys.shape
            batch_xs, batch_ys, available_examples = self.get_batch_by_random(factors_train, labels_train)
            # print("available_batch_size", available_batch_size)
            if len(available_examples) < self.model.batch_size:
                break
            # batch_xs = batch_xs.reshape([model.batch_size, model.n_step, model.n_inputs])
            # feed_dict = {model.xs: batch_xs, model.ys: batch_ys}
            # sess.run(model.train_op, feed_dict=feed_dict)
            feed_dict = {
                self.model.xs: batch_xs,
                self.model.ys: batch_ys,
                self.model.is_training: True,
            }

            # sess.run(model.train_op, feed_dict=feed_dict)
            _, cost, state, train_accuracy = sess.run(
                [self.model.train_op, self.model.cost, self.model.cell_final_state, self.model.accuracy_op]
                , feed_dict=feed_dict
            )

            if step % 20 == 0:
                # train_accuracy = np.mean(np.argmax(pred, 1) == np.argmax(batch_ys, 1))
                batch_xs, batch_ys, _ = self.get_batch_by_random(factors_validation, labels_validation)
                feed_dict = {
                    self.model.xs: batch_xs,
                    self.model.ys: batch_ys,
                    self.model.is_training: False,
                    # TODO: model.is_training should be False
                }
                test_accuracy = sess.run(self.model.accuracy_op, feed_dict=feed_dict)
                logger.info('%d/%d) train: %s validation: %s', step, training_iters, train_accuracy, test_accuracy)
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, step)

            step += 1

        return self.model

    def save_model(self):
        """
        将模型导出到文件
        :return:
        """
        saver = tf.train.Saver()
        folder_path = get_folder_path('my_net', create_if_not_found=False)
        if folder_path is None:
            raise ValueError('folder_path: "my_net" not exist')
        file_path = os.path.join(folder_path, f"save_net_{self.output_size}_{self.normalization_model}.ckpt")
        save_path = saver.save(self.session, file_path)
        logger.info("Save to path: %s", save_path)
        self.model_file_path = save_path
        return save_path

    def load_model_if_exist(self):
        """
        将模型导出到文件
        :return:
        """
        if self.model_file_path is not None and os.path.exists(self.model_file_path):
            saver = tf.train.Saver()
            save_path = saver.restore(self.session, self.model_file_path)
            logger.info("load from path: %s", save_path)
            return True
        else:
            return False

    def predict_test(self, md_df):
        # logger.info('开始预测')
        sess = self.session
        # saver = tf.train.Saver()
        # saver.restore(sess, f"my_net/save_net_{self.model.normalization_model}.ckpt")
        factors, labels = self.get_factors_with_labels(md_df)
        logger.info("批量样本测试")
        batch_xs, batch_ys, available_examples = self.get_batch_by_random(factors, labels)
        feed_dict = {
            self.model.xs: batch_xs,
            self.model.ys: batch_ys,
            self.model.is_training: False,
            # TODO: model.is_training should be False
        }
        pred = sess.run(tf.argmax(self.model.pred, 1), feed_dict)
        logger.info("accuracy: %.2f%%" % (sum(pred == np.argmax(batch_ys, axis=1)) / len(pred) * 100))
        logger.info("pred: \n%s\n%s", pred, np.argmax(batch_ys, axis=1))
        # logger.info("batch_ys \n%s", np.argmax(batch_ys, axis=1))

        logger.info("独立样本测试(与上面批量样本测试结果应保持一致)")
        # batch_xs, batch_ys, available_batch_size = self.get_batch_by_random(factors, labels)
        pred_all, label_all = [], []
        # for n in range(len(available_examples)):
        #     feed_dict = {
        #         self.model.xs: batch_xs[n:n + 1, :, :],
        #         self.model.ys: batch_ys[n:n + 1, :],
        #         self.model.is_training: False,
        #         # TODO: model.is_training should be False
        #     }
        #     pred = sess.run(tf.argmax(self.model.pred, 1), feed_dict)
        #     pred_all.extend(pred)
        # logger.info("accuracy: %.2f%%" % (sum(pred_all == np.argmax(batch_ys, axis=1)) / len(pred_all) * 100))
        # logger.info("pred: \n%s\n%s", np.array(pred_all), np.argmax(batch_ys, axis=1))
        for num, index in enumerate(available_examples):
            pred_mark = self.predict_latest(md_df.iloc[:index, :])
            pred_all.append(pred_mark)
            label_all.append(np.argmax(labels[index]))
            logger.debug("(%d %d) (%d %d)",
                         pred_mark, np.argmax(labels[index]), pred[num], np.argmax(batch_ys, axis=1)[num])

        pred_all, label_all = np.array(pred_all), np.array(label_all)
        logger.info("accuracy: %.2f%%" % (sum(pred_all == label_all) / len(pred_all) * 100))
        logger.info("pred: \n%s\n%s", pred_all, label_all)

    def predict_latest(self, md_df):
        factors = self.get_factors(md_df, tail_n=self.n_step)
        batch_xs = self.get_batch_xs(factors)
        feed_dict = {
            self.model.xs: batch_xs,
            self.model.is_training: False,
            # TODO: model.is_training should be False
        }
        pred_mark = self.session.run(tf.argmax(self.model.pred, 1), feed_dict)[0]
        return pred_mark

    def on_prepare_min1(self, md_df, context):
        if md_df is None:
            return

        if self.is_load_model_if_exist:
            # 加载模型
            is_load = self.load_model_if_exist()
        else:
            is_load = False

        if not is_load:
            # 训练模型
            self.train(md_df)
            self.predict_test(md_df)
            self.save_model()

    def on_min1(self, md_df, context):
        pred_mark = self.predict_latest(md_df)
        is_sell, is_holding, is_buy = pred_mark == 0, pred_mark == 1, pred_mark == 2
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
            self.logger.debug("%s %s    %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)
            pass

    def separate_train_validation(self, factors, labels):
        data_len = factors.shape[0]
        train_len = int(self.train_validation_rate * data_len)
        # 至少留给 validation 样本集一个 batch_size 的数量
        if train_len > (data_len - self.batch_size - self.n_step):
            train_len = data_len - self.batch_size - self.n_step

        factors_train, factors_validation = factors[:train_len, :], factors[train_len:, :]
        labels_train, labels_validation = labels[:train_len, :], labels[train_len:, :]
        return factors_train, factors_validation, labels_train, labels_validation


def _test_use(is_plot):
    from ibats_common.backend.mess import get_folder_path
    import os
    # 参数设置
    run_mode = RunMode.Backtest
    strategy_params = {'unit': 100}
    md_agent_params_list = [{
        'md_period': PeriodType.Min1,
        'instrument_id_list': ['RB'],
        'datetime_key': 'trade_date',
        'init_md_date_from': '1995-1-1',  # 行情初始化加载历史数据，供策略分析预加载使用
        'init_md_date_to': '2014-1-1',
        # 'C:\GitHub\IBATS_Common\ibats_common\example\ru_price2.csv'
        'file_path': os.path.abspath(os.path.join(
            get_folder_path('example', create_if_not_found=False), 'data', 'RB.csv')),
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
            'init_cash': 1000000,
            "calc_mode": CalcMode.Margin,
        }
        strategy_handler_param = {
            'date_from': '2014-1-1',  # 策略回测历史数据，回测指定时间段的历史行情
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
        from ibats_common.analysis.plot_db import show_order, show_cash_and_margin, show_rr_with_md
        show_order(stg_run_id)
        show_cash_and_margin(stg_run_id)
        show_rr_with_md(stg_run_id)

    return stg_run_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG, format=config.LOG_FORMAT)
    is_plot = True
    _test_use(is_plot)
