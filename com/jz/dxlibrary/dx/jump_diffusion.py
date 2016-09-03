#!/usr/bin/env python
# coding=utf-8


# jump diffusion

import numpy as np

from .simulation_class import simulation_class
from .sn_random_numbers import sn_random_numbers

class jump_diffusion(simulation_class):

    """
    基于Merton(1976)的jump diffusion模型来生成模拟路径
    """

    def __init__(self, name, mar_env, corr=False):
        super(jump_diffusion, self).__init__(name, mar_env, corr)

        try:
            self.lamb = mar_env.get_constant('lambda')
            self.mu = mar_env.get_constant('mu')
            self.delta = mar_env.get_constant('delta')

        except:
            print("Error parsing market environment.")


    def update(self, initial_value=None, volatility=None, lamb=None,
               mu=None, delta=None, final_date=None):

        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delta = delta

        self.instrument_values = None


    def generate_paths(self, fixed_seed=True, day_count=365.):

        if self.time_grid is None:
            self.generate_time_grid()

        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths[0] = self.initial_value

        # 这里sn1和sn2 代表了zt1和zt2
        # zt1和yt可能会相关，而zt2和yt是不相关的，所以sn2直接生成就好。sn1要特殊处理
        if not self.correlated:
            sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            sn1 = self.random_numbers

        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        #
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delta**2) - 1)
        short_rate = self.discount_curve.short_rate

        for t in range(1, len(self.time_grid)):
            if not self.correlated:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]

            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count

            # 泊松分布的随机数
            poi = np.random.poisson(self.lamb * dt, I)

            paths[t] = paths[t - 1] * (np.exp(
                (short_rate - rj - 0.5 * self.volatility**2) * dt +
                self.volatility * np.sqrt(dt) * ran) +
                (np.exp(self.mu + self.delta * sn2[t]) - 1) * poi)

        self.instrument_values = paths






