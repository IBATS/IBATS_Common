#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-8-8 下午12:28
@File    : utils.py
@contact : mmmaaaggg@163.com
@desc    : use_cup_only 需要优先于一切 import tensorflow 的语句被调用才能生效
"""


def show_device():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()


def _test_show_device():
    devices = show_device()
    print(type(devices), len(devices))
    for num, d in enumerate(devices):
        print(num, '\n', d)


def use_cup_only():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow.python.keras.backend as K

    import tensorflow as tf
    K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


if __name__ == '__main__':
    _test_show_device()
