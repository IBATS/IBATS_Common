#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-8-8 下午12:28
@File    : utils.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from tensorflow.python.client import device_lib
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def show_device():
    return device_lib.list_local_devices()


def _test_show_device():
    devices = show_device()
    print(type(devices), len(devices))
    for num, d in enumerate(devices):
        print(num, '\n', d)


def use_cup_only():
    ktf.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


if __name__ == '__main__':
    _test_show_device()
