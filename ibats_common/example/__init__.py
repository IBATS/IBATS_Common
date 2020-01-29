#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2018/11/7 10:42
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import ffn
import numpy as np
import pandas as pd
from ibats_utils.mess import copy_module_file_to, date_2_str, str_2_date, get_last, copy_file_to
from sklearn.model_selection import train_test_split
from ibats_common.backend.mess import get_report_folder_path
from ibats_common.analysis.plot import show_dl_accuracy
from ibats_common.analysis.summary import summary_release_2_docx
from ibats_common.backend.factor import get_factor
from ibats_common.backend.label import calc_label3
from ibats_common.common import ContextKey, Direction
from ibats_common.example.data import get_trade_date_series, get_delivery_date_series
from ibats_common.strategy import StgBase

logger = logging.getLogger(__name__)
FFN_VERSION = ffn.__version__


class AIStgBase(StgBase):

    def __init__(self, instrument_type, unit=1):
        super().__init__()
        self.unit = unit
        # 模型相关参数
        self.input_size = 38
        self.batch_size = 512
        self.n_step = 60
        self.output_size = 3
        self.n_hidden_units = 72
        self.lr = 0.006
        # 模型训练，及数据集相关参数
        self._model = None
        self._session = None
        self.train_validation_rate = 0.8
        self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = None, None, None, None
        self.label_func_max_rr = 0.0051
        self.label_func_min_rr = -0.0054
        self.max_future = 3
        self.predict_test_random_state = None
        self.n_epoch = 40
        self.max_loop_4_futher_train = 20
        self.retrain_period = 30  # 60 每隔60天重新训练一次，0 则为不进行重新训练
        self.validation_accuracy_base_line = 0.55  # 0.6    # 如果为 None，则不进行 validation 成功率检查
        self.over_fitting_train_acc = 0.95  # 过拟合训练集成功率，如果为None则不进行判断
        # 其他辅助信息
        self.trade_date_series = get_trade_date_series()
        self.delivery_date_series = get_delivery_date_series(instrument_type)
        self.tensorboard_verbose = 3
        # 模型保存路径相关参数
        # 是否使用现有模型进行操作，如果是记得修改以下下方的路径
        # enable_load_model_if_exist 将会在调用 self.load_model_if_exist 时进行检查
        # 如果该字段为 False，调用 load_model_if_exist 时依然可以传入参数的方式加载已有模型
        # 该字段与 self.load_model_if_exist 函数的 enable_load_model_if_exist参数是 “or” 的关系
        self.enable_load_model_if_exist = False
        if self.enable_load_model_if_exist:
            self.base_folder_path = folder_path = os.path.join(
                get_report_folder_path(self.stg_run_id), 'tf_saves_2019-06-27_16_24_34')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            self.base_folder_path = folder_path = os.path.join(
                get_report_folder_path(self.stg_run_id), f'tf_saves_{datetime_str}')
        # 备份文件
        file_path = copy_module_file_to(self.__class__, folder_path)
        self.logger.debug('文件已经备份到：%s', file_path)
        # 设置模型备份目录
        self.model_folder_path = model_folder_path = os.path.join(folder_path, 'model_tfls')
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path, exist_ok=True)
        # 设置日至备份目录
        self.tensorboard_dir = tensorboard_dir = os.path.join(folder_path, f'tensorboard_logs')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.trade_date_last_train = None
        self.trade_date_acc_list = defaultdict(lambda: [0.0, 0.0])
        self.do_nothing_on_min_bar = False  # 仅供调试使用
        # 用于记录 open,high,low,close,amount 的 column 位置
        self.ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]

    @property
    def session(self):
        return self.get_session()

    def get_session(self, renew=False, close_last_one_if_renew=True):
        import tensorflow as tf
        if renew or self._session is None:
            if self.model is None:
                raise ValueError('model 需要先于 session 被创建')
            if close_last_one_if_renew and self._session is not None:
                self._session.close()
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session

    @property
    def model(self):
        """
        获取模型对象
        2019-07-05 去除  -> tflearn.models.DNN hint 该语句将会导致动态加载 class 时抛出错误
        :return:
        """
        return self.get_model()

    def get_model(self, rebuild_model=False):
        """
        获取模型对象，默认不重新构造
        2019-07-05 去除  -> tflearn.models.DNN hint 该语句将会导致动态加载 class 时抛出错误
        :param rebuild_model:
        :return:
        """
        import tensorflow as tf
        if self._model is None or rebuild_model:
            self.logger.info('重新构建模型')
            tf.reset_default_graph()
            self._model = self._build_model()
        return self._model

    def get_factor_array(self, md_df: pd.DataFrame, tail_n=None):
        if tail_n is not None:
            md_df = md_df.tail(tail_n)
        df = md_df[~md_df['close'].isnull()]

        factors = df.fillna(0).to_numpy()
        if self.input_size is None or self.input_size != factors.shape[1]:
            self.input_size = factors.shape[1]
            self.n_hidden_units = self.input_size * 2
            self.logger.info("set input_size: %d", self.input_size)
            self.logger.info("set n_hidden_units: %d", self.n_hidden_units)

        return factors, df.index

    def get_x_y(self, factor_df):
        factors, trade_date_index = self.get_factor_array(factor_df)
        price_arr = factors[:, 0]
        self.input_size = factors.shape[1]
        # ys_all = self.calc_y_against_future_data(price_arr, -0.01, 0.01)
        # ys_all = self.calc_y_against_future_data(price_arr, -self.classify_wave_rate, self.classify_wave_rate)
        ys_all = calc_label3(price_arr, self.label_func_min_rr, self.label_func_max_rr,
                             max_future=self.max_future, one_hot=True)
        range_from = self.n_step
        range_to = factors.shape[0]

        xs = np.zeros((range_to - range_from, self.n_step, self.input_size))
        for num, index in enumerate(range(range_from, range_to)):
            xs[num, :, :] = factors[(index - self.n_step):index, :]

        ys = ys_all[range_from:range_to, :]

        return xs, ys, trade_date_index[range_from:range_to]

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

    def _build_model(self):
        # Network building
        # net = tflearn.input_data([None, self.n_step, self.input_size])
        # ...
        # Training
        # _model = tflearn.DNN(net, tensorboard_verbose=self.tensorboard_verbose,
        #                      checkpoint_path=self.checkpoint_path,
        #                      tensorboard_dir=self.tensorboard_dir)
        # return _model
        raise NotImplementedError()

    def train(self, factor_df_dic: dict, predict_test_random_state):
        import tflearn
        factor_df = factor_df_dic[1]

        trade_date_from_str, trade_date_to_str = date_2_str(factor_df.index[0]), date_2_str(factor_df.index[-1])
        # xs_train, xs_validation, ys_train, ys_validation = self.separate_train_validation(xs, ys)
        if self.predict_test_random_state is None:
            random_state = predict_test_random_state
        else:
            random_state = self.predict_test_random_state

        # 利用生成数据做训练数据集，只用原始数据中的 validation 部分做验证集
        arr_list, xs_validation, ys_validation = [], None, None
        for adj_factor, factor_df in factor_df_dic.items():
            xs, ys, _ = self.get_x_y(factor_df)
            xs_train_tmp, xs_validation_tmp, ys_train_tmp, ys_validation_tmp = train_test_split(
                xs, ys, test_size=0.2, random_state=random_state)
            arr_list.append([xs_train_tmp, ys_train_tmp])
            # xs_train, xs_validation = xs_train_tmp, xs_validation_tmp
            # ys_train, ys_validation = ys_train_tmp, ys_validation_tmp
            if adj_factor == 1:
                xs_validation, ys_validation = xs_validation_tmp, ys_validation_tmp

        self.xs_train = xs_train = np.vstack([_[0] for _ in arr_list])
        self.ys_train = ys_train = np.vstack([_[1] for _ in arr_list])

        sess = self.get_session(renew=True)
        train_acc, val_acc = 0, 0
        with sess.as_default():
            # with tf.Graph().as_default():
            # self.logger.debug('sess.graph:%s tf.get_default_graph():%s', sess.graph, tf.get_default_graph())
            self.logger.debug('[%d], xs_train %s, ys_train %s, xs_validation %s, ys_validation %s, [%s, %s]',
                              random_state, xs_train.shape, ys_train.shape, xs_validation.shape, ys_validation.shape,
                              trade_date_from_str, trade_date_to_str)
            max_loop = self.max_loop_4_futher_train
            for num in range(max_loop):
                if num == 0:
                    n_epoch = self.n_epoch
                else:
                    n_epoch = self.n_epoch // max_loop

                self.logger.info('[%d]第 %d/%d 轮训练，开始 [%s, %s] n_epoch=%d', random_state, num + 1, max_loop,
                                 trade_date_from_str, trade_date_to_str, n_epoch)
                run_id = f'{trade_date_to_str}_{xs_train.shape[0]}[{predict_test_random_state}]' \
                         f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                tflearn.is_training(True)
                self.model.fit(
                    xs_train, ys_train, validation_set=(xs_validation, ys_validation),
                    show_metric=True, batch_size=self.batch_size, n_epoch=n_epoch,
                    run_id=run_id)
                tflearn.is_training(False)

                result = self.model.evaluate(xs_train, ys_train, batch_size=self.batch_size)
                # self.logger.info("train accuracy: %.2f%%" % (result[0] * 100))
                train_acc = result[0]
                result = self.model.evaluate(xs_validation, ys_validation, batch_size=self.batch_size)
                val_acc = result[0]
                self.logger.info("[%d]第 %d/%d 轮训练，[%s - %s]，训练集准确率(train_acc)：%.2f%%， 样本外准确率(val_acc): %.2f%%",
                                 random_state, num + 1, max_loop, trade_date_from_str, trade_date_to_str,
                                 train_acc * 100, val_acc * 100)
                if self.over_fitting_train_acc is not None and train_acc > self.over_fitting_train_acc:
                    self.logger.warning('[%d]第 %d/%d 轮训练，训练集精度 %.2f%% > %.2f%% 可能存在过拟合 [%s, %s]',
                                        random_state, num + 1, max_loop, train_acc * 100,
                                        self.over_fitting_train_acc * 100,
                                        trade_date_from_str, trade_date_to_str)
                    break
                if self.validation_accuracy_base_line is not None:
                    if result[0] > self.validation_accuracy_base_line:
                        break
                    elif num < max_loop - 1:
                        self.logger.warning('[%d]第 %d/%d 轮训练，[%s - %s]，样本外训练准确率 %.2f%% < %.0f%%，继续训练',
                                            random_state, num + 1, max_loop, trade_date_from_str, trade_date_to_str,
                                            val_acc * 100, self.validation_accuracy_base_line * 100)
                else:
                    break

        self.trade_date_last_train = str_2_date(trade_date_to_str)
        return train_acc, val_acc

    def valid_model_acc(self, factor_df: pd.DataFrame):
        xs, ys, _ = self.get_x_y(factor_df)
        trade_date_from_str, trade_date_to_str = date_2_str(factor_df.index[0]), date_2_str(factor_df.index[-1])
        random_state = self.predict_test_random_state
        xs_train, xs_validation, ys_train, ys_validation = train_test_split(
            xs, ys, test_size=0.2, random_state=random_state)
        self.logger.debug('random_state=%d, xs_train %s, ys_train %s, xs_validation %s, ys_validation %s, [%s, %s]',
                          random_state, xs_train.shape, ys_train.shape, xs_validation.shape, ys_validation.shape,
                          trade_date_from_str, trade_date_to_str)
        sess = self.get_session(renew=True)
        with sess.as_default():
            # with tf.Graph().as_default():
            # self.logger.debug('sess.graph:%s tf.get_default_graph():%s', sess.graph, tf.get_default_graph())
            result = self.model.evaluate(xs_validation, ys_validation, batch_size=self.batch_size)
            val_acc = result[0]
            result = self.model.evaluate(xs_train, ys_train, batch_size=self.batch_size)
            train_acc = result[0]
            self.logger.info("[%s - %s] 训练集准确率: %.2f%%", trade_date_from_str, trade_date_to_str, train_acc * 100)
            self.logger.info("[%s - %s] 验证集准确率: %.2f%%", trade_date_from_str, trade_date_to_str, val_acc * 100)
        return train_acc, val_acc

    def save_model(self, trade_date):
        """
        将模型导出到文件
        :param trade_date:
        :return:
        """
        folder_path = os.path.join(self.model_folder_path, date_2_str(trade_date))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(
            folder_path,
            f"model_{int(self.label_func_min_rr * 10000)}_{int(self.label_func_max_rr * 10000)}.tfl")
        self.model.save(file_path)
        self.logger.info("模型训练截止日期： %s 保存到: %s", self.trade_date_last_train, file_path)
        return file_path

    def load_model_if_exist(self, trade_date, enable_load_model_if_exist=False):
        """
        将模型导出到文件
        目录结构：
        tf_saves_2019-06-05_16_21_39
          *   model_tfls
          *       *   2012-12-31
          *       *       *   checkpoint
          *       *       *   model_-54_51.tfl.data-00000-of-00001
          *       *       *   model_-54_51.tfl.index
          *       *       *   model_-54_51.tfl.meta
          *       *   2013-02-28
          *       *       *   checkpoint
          *       *       *   model_-54_51.tfl.data-00000-of-00001
          *       *       *   model_-54_51.tfl.index
          *       *       *   model_-54_51.tfl.meta
          *   tensorboard_logs
          *       *   2012-12-31_496[1]_20190605_184316
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
          *       *   2013-02-28_496[1]_20190605_184716
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
        :param enable_load_model_if_exist:
        :param trade_date:
        :return:
        """
        if self.enable_load_model_if_exist or enable_load_model_if_exist:
            # 获取小于等于当期交易日的最大的一个交易日对应的文件名
            min_available_date = str_2_date(trade_date) - timedelta(days=self.retrain_period)
            self.logger.debug('尝试加载现有模型，[%s - %s] %d 天', min_available_date, trade_date, self.retrain_period)
            date_file_path_pair_list = [_ for _ in self.get_date_file_path_pair_list() if _[0] >= min_available_date]
            if len(date_file_path_pair_list) > 0:
                # 按日期排序
                date_file_path_pair_list.sort(key=lambda x: x[0])
                # 获取小于等于当期交易日的最大的一个交易日对应的文件名
                # file_path = get_last(date_file_path_pair_list, lambda x: x[0] <= trade_date, lambda x: x[1])
                trade_date = str_2_date(trade_date)
                ret = get_last(date_file_path_pair_list, lambda x: x[0] <= trade_date)
                if ret is not None:
                    key, folder_path, predict_test_random_state = ret
                    if folder_path is not None:
                        model = self.get_model(rebuild_model=True)  # 这句话是必须的，需要实现建立模型才可以加载
                        model.load(folder_path)
                        self.trade_date_last_train = key
                        self.predict_test_random_state = predict_test_random_state
                        self.logger.info("加载模型成功。trade_date_last_train: %s load from path: %s", key, folder_path)
                        return True

        return False

    def predict_test(self, md_df):
        self.logger.info('开始预测')
        # sess = self.session
        if self.xs_train is None:
            xs, ys, _ = self.get_x_y(md_df)
            xs_train, xs_validation, ys_train, ys_validation = train_test_split(xs, ys, test_size=0.2, random_state=1)
            self.xs_train, self.xs_validation, self.ys_train, self.ys_validation = xs_train, xs_validation, ys_train, ys_validation

        with self.session.as_default() as sess:
            real_ys = np.argmax(self.ys_validation, axis=1)

            # self.logger.info("批量预测2")
            # pred_ys = np.argmax(self.model.predict_label(self.xs_validation), axis=1) 与 evaluate 结果刚好相反
            # 因此不使用 predict_label 函数
            # pred_ys = np.argmax(self.model.predict(self.xs_validation), axis=1)
            # self.logger.info("模型训练基准日期：%s，validation accuracy: %.2f%%",
            #             self.trade_date_last_train, sum(pred_ys == real_ys) / len(pred_ys) * 100)
            # self.logger.info("pred/real: \n%s\n%s", pred_ys, real_ys)

            self.logger.info("独立样本预测(predict_latest)")
            pred_ys = []
            for idx, y in enumerate(self.ys_validation):
                x = self.xs_validation[idx:idx + 1, :, :]
                pred_y = self.model.predict(x)
                pred_ys.extend(np.argmax(pred_y, axis=1))

            pred_ys = np.array(pred_ys)
            self.logger.info("模型训练基准日期：%s，validation accuracy: %.2f%%",
                             self.trade_date_last_train, sum(pred_ys == real_ys) / len(pred_ys) * 100)
            # self.logger.info("pred: \n%s\n%s", pred_ys, real_ys)

    def predict_latest(self, factor_df):
        """
        计算最新一个 X，返回分类结果
        二分类，返回 0 未知 / 1 下跌 / 2 上涨
        三分类，返回 0 震荡 / 1 下跌 / 2 上涨
        :param factor_df:
        :return:
        """
        factors, _ = self.get_factor_array(factor_df, tail_n=self.n_step)
        x = self.get_batch_xs(factors)
        pred_y = np.argmax(self.model.predict(x), axis=1)[-1]
        # is_buy, is_sell = pred_mark == 1, pred_mark == 0
        # return is_buy, is_sell
        return pred_y

    def on_prepare_min1(self, md_df, context):
        if md_df is None:
            return
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)
        self.load_train_test(indexed_df, enable_load_model=self.enable_load_model_if_exist)

    def load_train_test(self, indexed_df, enable_load_model, rebuild_model=False, enable_train_if_load_not_suss=True,
                        enable_train_even_load_succ=False, enable_test=False):
        if rebuild_model:
            self.get_model(rebuild_model=True)

        trade_date = str_2_date(indexed_df.index[-1])
        # 加载模型
        if enable_load_model:
            is_load = self.load_model_if_exist(trade_date)
        else:
            is_load = False

        if enable_train_even_load_succ or (enable_train_if_load_not_suss and not is_load):
            factor_df_dic = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                                       trade_date_series=self.trade_date_series,
                                       delivery_date_series=self.delivery_date_series, do_multiple_factors=True)
            factor_df = factor_df_dic[1]
            num = 0
            while True:
                num += 1
                if num > 1:
                    self.get_model(rebuild_model=True)
                # 训练模型
                train_acc, val_acc = self.train(factor_df_dic, predict_test_random_state=num)
                if self.over_fitting_train_acc is not None and train_acc > self.over_fitting_train_acc:
                    self.logger.warning('第 %d 次训练，训练集精度 train_acc=%.2f%% 过高，可能存在过拟合，重新采样训练',
                                        num, train_acc * 100)
                    continue
                if self.validation_accuracy_base_line is not None:
                    if val_acc < self.validation_accuracy_base_line:
                        self.logger.warning('第 %d 次训练，训练结果不及预期，重新采样训练', num)
                        continue
                    # elif train_acc - val_acc > 0.15 and val_acc < 0.75:
                    #     self.logger.warning('第 %d 次训练，train_acc=%.2f%%, val_acc=%.2f%% 相差大于 15%% 且验证集正确率小于75%%，重新采样训练',
                    #                    num, train_acc * 100, val_acc * 100)
                    #     continue
                    else:
                        break
                else:
                    break

            self.save_model(trade_date)
            self.trade_date_last_train = trade_date
        else:
            factor_df = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                                   trade_date_series=self.trade_date_series,
                                   delivery_date_series=self.delivery_date_series)
            train_acc, val_acc = self.valid_model_acc(factor_df)

        self.trade_date_acc_list[trade_date] = [train_acc, val_acc]

        # enable_test 默认为 False
        # self.valid_model_acc(factor_df) 以及完全取代 self.predict_test
        # self.predict_test 仅用于内部测试使用
        if enable_test:
            self.predict_test(factor_df)

        return factor_df

    def on_min1(self, md_df, context):
        if self.do_nothing_on_min_bar:  # 仅供调试使用
            return

        # 数据整理
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        indexed_df.index = pd.DatetimeIndex(indexed_df.index)
        # 获取最新交易日
        trade_date = str_2_date(indexed_df.index[-1])
        days_after_last_train = (trade_date - self.trade_date_last_train).days
        if self.retrain_period is not None and 0 < self.retrain_period < days_after_last_train:
            # 重新训练
            self.logger.info('当前日期 %s 距离上一次训练 %s 已经过去 %d 天，重新训练',
                             trade_date, self.trade_date_last_train, days_after_last_train)
            factor_df = self.load_train_test(indexed_df, rebuild_model=True,
                                             enable_load_model=self.enable_load_model_if_exist)
        else:
            factor_df = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                                   trade_date_series=self.trade_date_series,
                                   delivery_date_series=self.delivery_date_series)

        # 预测
        pred_mark = self.predict_latest(factor_df)
        is_holding, is_buy, is_sell = pred_mark == 0, pred_mark == 1, pred_mark == 2
        # self.logger.info('%s is_buy=%s, is_sell=%s', trade_date, str(is_buy), str(is_sell))
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
                self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

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
                self.logger.debug("%s %s     %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

        if is_holding:
            self.logger.debug("%s %s * * %.2f holding", self.trade_agent.curr_timestamp, instrument_id, close)

    def on_min1_release(self, md_df):
        """
        增加模型对未来数据预测成功率走势图展示
        :param md_df:
        :return:
        """
        if md_df is None or md_df.shape[0] == 0:
            self.logger.warning('md_df is None or shape[0] == 0')
            return
        else:
            self.logger.debug('md_df.shape= %s', md_df.shape)

        # 获取各个模型训练时间点及路径
        date_file_path_pair_list = self.get_date_file_path_pair_list()
        if len(date_file_path_pair_list) > 0:
            # 按日期排序
            date_file_path_pair_list.sort(key=lambda x: x[0])

        # 建立数据集
        indexed_df = md_df.set_index('trade_date').drop('instrument_type', axis=1)
        trade_date_end = indexed_df.index[-1]
        factor_df = get_factor(indexed_df, ohlcav_col_name_list=self.ohlcav_col_name_list,
                               trade_date_series=self.trade_date_series,
                               delivery_date_series=self.delivery_date_series)
        xs, ys_onehot, trade_date_index = self.get_x_y(factor_df)
        ys = np.argmax(ys_onehot, axis=1)
        data_len = len(trade_date_index)
        if data_len == 0:
            self.logger.warning('ys 长度为0，请检查是否存在数据错误')
            return
        trade_date2_list = [_[0] for _ in date_file_path_pair_list][1:]
        trade_date2_list.append(None)
        # 预测结果
        self.logger.info("按日期分段验证检验预测结果")
        pred_ys_tot, real_ys_tot, img_meta_dic_list = [], [], []
        # 根据模型 trade_date_last_train 进行分段预测，并将结果记录到 pred_ys
        for num, ((trade_date_last_train, file_path, predict_test_random_state), trade_date_next) in enumerate(zip(
                date_file_path_pair_list, trade_date2_list)):
            # 以模型训练日期为基准，后面的数据作为验证集数据（样本外数据）
            # 获取有效的日期范围 from - to
            range_from_arr = trade_date_index >= pd.to_datetime(trade_date_last_train)
            range_from_len = len(range_from_arr)
            if range_from_len == 0:  # range_from_len 应该与 trade_date_list_count 等长度，所以这个条件应该永远不会满足
                self.logger.error('总共%d条数据，%s 开始后面没有可验证数据', data_len, trade_date_last_train)
                continue
            true_count = sum(range_from_arr)
            self.logger.debug("len(range_from)=%d, True Count=%d", len(range_from_arr), true_count)
            if true_count == 0:
                self.logger.warning('总共%d条数据，%s 开始后面没有可验证数据', data_len, trade_date_last_train)
                continue
            # 自 trade_date_last_train 起的所有有效日期
            trade_date_list_sub = trade_date_index[range_from_arr]

            # 获取 in_range，作为 range_from, range_to 的交集
            if trade_date_next is None:
                in_range_arr = None
                in_range_count = true_count
            else:
                in_range_arr = trade_date_list_sub < pd.to_datetime(trade_date_next)
                in_range_count = sum(in_range_arr)
                if in_range_count == 0:
                    self.logger.warning('总共%d条数据，[%s - %s) 之间没有可用数据',
                                        data_len, trade_date_last_train, trade_date_next)
                    continue
                else:
                    self.logger.debug('总共%d条数据，[%s - %s) 之间有 %d 条数据将被验证 model path:%s',
                                      data_len, trade_date_last_train, trade_date_next, in_range_count, file_path)

            # 获取当前时段对应的 xs
            # 进行验证时，对 range_from 开始的全部数据进行预测，按照 range_to 为分界线分区着色显示
            xs_sub, real_ys = xs[range_from_arr, :, :], ys[range_from_arr]
            close_df = indexed_df.loc[trade_date_list_sub, 'close']

            # 加载模型
            is_load = self.load_model_if_exist(trade_date_last_train, enable_load_model_if_exist=True)
            if not is_load:
                self.logger.error('%s 模型加载失败：%s', trade_date_last_train, file_path)
                continue
            # 预测
            pred_ys_one_hot = self.model.predict(xs_sub)
            pred_ys = np.argmax(pred_ys_one_hot, axis=1)
            if in_range_arr is not None and in_range_count > 0:
                pred_ys_tot.extend(pred_ys[in_range_arr])
            else:
                pred_ys_tot.extend(pred_ys)

            # 为每一个时段单独验证成功率，以当前模型为基准，验证后面全部历史数据成功率走势
            if trade_date_next is None:
                split_point_list = None
            else:
                split_point_list = [close_df.index[0], trade_date_next, close_df.index[-1]]
            base_line_list = self.trade_date_acc_list[trade_date_last_train]
            img_file_path = show_dl_accuracy(real_ys, pred_ys, close_df, split_point_list,
                                             base_line_list=base_line_list)
            img_meta_dic_list.append({
                'img_file_path': img_file_path,
                'trade_date_last_train': trade_date_last_train,
                'module_file_path': file_path,
                'predict_test_random_state': predict_test_random_state,
                'split_point_list': split_point_list,
                'in_range_count': in_range_count,
                'trade_date_end': trade_date_end,
            })

        pred_ys_tot = np.array(pred_ys_tot)
        trade_date_last_train_first = pd.to_datetime(date_file_path_pair_list[0][0])
        split_point_list = [_[0] for _ in date_file_path_pair_list]
        split_point_list.append(trade_date_index[-1])
        # 获取 real_ys
        real_ys = ys[trade_date_index >= trade_date_last_train_first]
        close_df = indexed_df.loc[trade_date_index[trade_date_index >= trade_date_last_train_first], 'close']
        img_file_path = show_dl_accuracy(real_ys, pred_ys_tot, close_df, split_point_list)

        img_meta_dic_list.append({
            'img_file_path': img_file_path,
            'trade_date_last_train': trade_date_last_train_first,
            'module_file_path': date_file_path_pair_list[0][1],
            'predict_test_random_state': date_file_path_pair_list[0][2],
            'split_point_list': split_point_list,
            'in_range_count': close_df.shape[0],
            'trade_date_end': trade_date_end,
        })
        is_output_docx = True
        if is_output_docx:
            title = f"[{self.stg_run_id}] predict accuracy trend report"
            file_path = summary_release_2_docx(title, img_meta_dic_list)
            copy_file_to(file_path, self.base_folder_path)

    def get_date_file_path_pair_list(self):
        """
        目录结构：
        tf_saves_2019-06-05_16_21_39
          *   model_tfls
          *       *   2012-12-31
          *       *       *   checkpoint
          *       *       *   model_-54_51.tfl.data-00000-of-00001
          *       *       *   model_-54_51.tfl.index
          *       *       *   model_-54_51.tfl.meta
          *       *   2013-02-28
          *       *       *   checkpoint
          *       *       *   model_-54_51.tfl.data-00000-of-00001
          *       *       *   model_-54_51.tfl.index
          *       *       *   model_-54_51.tfl.meta
          *   tensorboard_logs
          *       *   2012-12-31_496[1]_20190605_184316
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
          *       *   2013-02-28_496[1]_20190605_184716
          *       *       *   events.out.tfevents.1559731396.mg-ubuntu64
        :return:
        """
        # 获取全部文件名
        pattern = re.compile(r'model_[-]?\d+_\d+.tfl')
        date_file_path_pair_list, model_name_set = [], set()
        for folder_name in os.listdir(self.model_folder_path):
            folder_path = os.path.join(self.model_folder_path, folder_name)
            if os.path.isdir(folder_path):
                try:
                    # 获取 trade_date_last_train
                    key = str_2_date(folder_name)
                    for file_name in os.listdir(folder_path):
                        # 对下列有效文件名，匹配结果："model_-54_51.tfl"
                        # model_-54_51.tfl.data-00000-of-00001
                        # model_-54_51.tfl.index
                        # model_-54_51.tfl.meta
                        m = pattern.search(file_name)
                        if m is None:
                            continue
                        model_name = m.group()
                        if key in model_name_set:
                            continue
                        model_name_set.add(key)
                        # 获取 model folder_path
                        file_path = os.path.join(folder_path, model_name)
                        # 获取 predict_test_random_state
                        for log_folder_path in os.listdir(self.tensorboard_dir):
                            if log_folder_path.find(folder_name) == 0:
                                predict_test_random_state = int(log_folder_path.split('[')[1].split(']')[0])
                                break
                        else:
                            predict_test_random_state = None

                        date_file_path_pair_list.append([key, file_path, predict_test_random_state])
                except:
                    pass

        return date_file_path_pair_list
