#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-5-24 下午3:16
@File    : label.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calc_label2(value_arr: np.ndarray, min_rr: float, max_rr: float, one_hot=True, dtype='float32'):
    """
    根据时间序列数据 pct_arr 计算每一个时点目标标示 0 1 2
    计算方式：
    当某一点未来波动首先 < 下届 min_pct，则标记为： [0, 1, 0]
    当某一点未来波动首先 > 上届 max_pct，则标记为： [0, 0, 1]
    :param value_arr:
    :param min_rr:
    :param max_rr:
    :param one_hot:
    :param dtype:
    :return:
    """
    from tensorflow.keras.utils import to_categorical
    value_arr[np.isnan(value_arr)] = 0
    arr_len = value_arr.shape[0]
    # target_arr = np.zeros((arr_len, 2))
    label_arr = np.zeros(arr_len, dtype=dtype)
    for i in range(arr_len):
        base = value_arr[i]
        for j in range(i + 1, arr_len):
            result = value_arr[j] / base - 1
            if result < min_rr:
                label_arr[i] = 1
                break
            elif result > max_rr:
                label_arr[i] = 2
                break
    if one_hot:
        # target_arr = pd.get_dummies(label_arr)
        target_arr = to_categorical(label_arr, num_classes=3, dtype=dtype)
    else:
        target_arr = label_arr

    return target_arr


def _test_calc_label2(show_plt=True):
    import matplotlib.pyplot as plt
    i_s = np.arange(0, 20, 0.1)
    price_arr = np.cos(i_s) + 3  # cos(x) + 5
    one_hot = False
    labels = calc_label2(price_arr, -0.1, 0.1, one_hot=one_hot)
    if show_plt:
        plt.plot(i_s, labels, 'r',
                 i_s, price_arr, 'b--',
                 )
        plt.show()

    one_hot = True
    labels = calc_label2(price_arr, -0.1, 0.1, one_hot=one_hot)
    if show_plt:
        plt.plot(i_s, np.argmax(labels, axis=1), 'r',
                 i_s, price_arr, 'b--',
                 )
        plt.show()


def calc_label3(value_arr: np.ndarray, min_rr: float, max_rr: float, max_future: int, one_hot=True, dtype='float32'):
    """
    根据时间序列数据 pct_arr 计算每一个时点目标标示 -1 0 1
    计算方式：
    max_future 内未突破min_pct 或 max_pct，则标记为:  [1, 0, 0]
    当某一点未来波动首先 < 下届 min_pct，则标记为:      [0, 1, 0]
    当某一点未来波动首先 > 上届 max_pct，则标记为:      [0, 0, 1]
    :param value_arr:
    :param min_rr:
    :param max_rr:
    :param max_future:
    :param one_hot:
    :param dtype:
    :return:
    """
    from tensorflow.keras.utils import to_categorical
    value_arr[np.isnan(value_arr)] = 0
    arr_len = value_arr.shape[0]
    # target_arr = np.zeros((arr_len, 2))
    label_arr = np.zeros(arr_len, dtype=dtype)
    for i in range(arr_len):
        base = value_arr[i]
        for j in range(i + 1, arr_len):
            result = value_arr[j] / base - 1
            if result < min_rr:
                label_arr[i] = 1
                break
            elif result > max_rr:
                label_arr[i] = 2
                break
            elif max_future is not None and j-i >= max_future:
                # 超过最大检测日期，则被认为是震荡势
                # label_arr[i] = 0
        # target_arr = pd.get_dummies(label_arr)
                break
    if all(label_arr == 0):
        logger.warning("当期数组长度 %d, min_rr=%f, max_rr=%f, max_future=%f， 标记结果全部为0",
                       arr_len, min_rr, max_rr, max_future)

    if one_hot:
        target_arr = to_categorical(label_arr, num_classes=3, dtype=dtype)
    else:
        target_arr = label_arr

    return target_arr


def _test_calc_label3(show_plt=True):
    import matplotlib.pyplot as plt
    i_s = np.arange(0, 20, 0.1)
    price_arr = np.cos(i_s) + 3  # cos(x) + 5
    one_hot = False
    labels = calc_label3(price_arr, -0.1, 0.1, max_future=5, one_hot=one_hot)
    if show_plt:
        plt.plot(i_s, labels, 'r',
                 i_s, price_arr, 'b--',
                 )
        plt.show()

    one_hot = True
    labels = calc_label3(price_arr, -0.1, 0.1, max_future=5, one_hot=one_hot)
    if show_plt:
        plt.plot(i_s, np.argmax(labels, axis=1), 'r',
                 i_s, price_arr, 'b--',
                 )
        plt.show()


if __name__ == "__main__":
    _test_calc_label2()
    # _test_calc_label3()
