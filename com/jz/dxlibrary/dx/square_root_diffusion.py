#!/usr/bin/env python
# coding=utf-8

# square root diffusion

import numpy as np

from .simulation_class import simulation_class
from .sn_random_numbers import sn_random_numbers


class square_root_diffusion(simulation_class):
    """
    用来基于Cox-Ingersoll-Ross(1985)平方跟扩散过程来生成模拟路径
    """

    def __init__(self, name, mar_env, corr=None):
        super(square_root_diffusion, self).__init__(name, mar_env, corr)

        try:
            self.kappa = mar_env.get_constant('kappa')
            self.theta = mar_env.get_constant('theta')

        except:
            print("Error parsing market environment.")

    def update(self, initial_value=None, volatility=None, kappa=None,
               theta=None, final_date=None):

        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta

        self.instrument_values = None

    def generate_paths(self, fixed_seed=True, day_count=365.):

        if self.time_grid is None:
            self.generate_time_grid()

        M = len(self.time_grid)
        I = self.paths
        # x+部分
        paths = np.zeros((M, I))
        # x_hat 部分
        paths_ = np.zeros_like(paths)

        paths[0] = self.initial_value
        paths_[0] = self.initial_value

        if not self.correlated:
            # 没有相关性
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # 有相关性，需要在mar_env中给出描述
            rand = self.random_numbers

        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            paths_[t] = (paths_[t - 1] +
                         self.kappa * (self.theta - np.maximum(0, paths_[t - 1, :])) * dt +
                         np.sqrt(np.maximum(0, paths_[t - 1, :])) *
                         self.volatility * np.exp(dt) * ran)

            paths[t] = np.maximum(0, paths_[t])

        self.instrument_values = paths
