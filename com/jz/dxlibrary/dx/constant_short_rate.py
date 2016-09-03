#!/usr/bin/env python
# coding=utf-8

# 贴现因子计算

from .get_year_delta import *


class constant_short_rate(object):
    """
    用于short rate的discount
    """

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate

        if short_rate < 0:
            raise ValueError("Short Rate must be positive!")


    def get_discount_factores(self, date_list, dtobjects=True):
        """

        :param date_list:
        :param dtobjects: dtobjects is True, 将date_list转换为年份数
                否则，date_list必须是年分数
        :return:
        """
        if dtobjects is True:
            dlist = get_year_delta(date_list)

        else:
            dlist = np.array(date_list)

        # e(-rT)
        dflist = np.exp(self.short_rate * np.sort(-dlist))

        return np.array((date_list, dflist)).T















