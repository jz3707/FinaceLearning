#!/usr/bin/env python
# coding=utf-8


from .valuation_class import valuation_class
import numpy as np


class valuation_mcs_european(valuation_class):

    """
    对任意支付的欧式期权进行定价
    标的资产和期权之间的任意关系都可以模拟出来
    """

    def generate_payoff(self, fixed_seed=False):
        """
        给了标的价值，计算到期日的价值
        :param fixed_seed:
        :return:
        """

        # 期货就是没有行权极值的。
        try:
            stride = self.stride
        except:
            pass

        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid

        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print("Maturity date is not in time_grid of underlying.")
















