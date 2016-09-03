#!/usr/bin/env python
# coding=utf-8

# 获得日期的差异

# import datetime as dt

import numpy as np


def get_year_delta(date_list, day_count=365.):
    """
    返回日期间隔比例的与年的比例
    :param date_list:
    :param day_count:
    :return:
    """

    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]

    return np.array(delta_list)

