#!/usr/bin/env python
# cofing=utf-8


import numpy as np
import pandas as pd

class simulation_class(object):
    """
    模拟类的基础方法
    """

    def __init__(self, name, mar_env, corr):

        try:
            self.name = name
            # 开始日期
            self.pricing_date = mar_env.pricing_date
            # 这个是什么的初始值
            self.initial_value = mar_env.get_constant('initial_value')
            # 波动率
            self.volatility = mar_env.get_constant('volatility')
            # 到期日
            self.final_date = mar_env.get_constant('final_date')
            self.currency = mar_env.get_constant('currency')
            # 年度的，季度，月度
            self.frequency = mar_env.get_constant('frequency')
            # 生成的路径的数量
            self.paths = mar_env.get_constant('paths')
            # 贴现率
            self.discount_curve = mar_env.get_curve('discount_curve')

            try:
                self.time_grid = mar_env.get_list('time_grid')
            except:
                self.time_grid = None

            try:
                self.sepcial_date = mar_env.get_list('sepcial_date')
            except:
                self.sepcial_date = []

            self.instrument_values = None
            # 是否有相关性，如果有相关性需要做Cholesky分解
            self.correlated = corr
            if corr is True:
                # 有相关性的时候要给出这三个集合。
                self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
                self.rn_set = mar_env.get_list('rn_set')[self.name]
                self.random_numbers = mar_env.get_list('random_numbers')
        except:
            print("Error parsing market environment!")





    def generate_time_grid(self):
        """
        如果mar_env给出了就直接用，如果没有给出需要生成
        :return:
        """
        start = self.pricing_date
        end = self.final_date

        # pandas.date_range, freq=B, business day, W, week, M, month
        time_grid = pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)

        # 处理special_date
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)

        if len(self.sepcial_date) > 0:
            time_grid.extend(self.sepcial_date)
            # 可能会有重复
            time_grid = list(set(time_grid))
            time_grid.sort()

        self.time_grid = np.array(time_grid)


    def get_instrument_values(self, fixed_seed=True):
        """
        得到工具的价值
        也就是模拟
        :param fixed_seed:
        :return:
        """

        if self.instrument_values is None:
            # self.generate_paths 派生类生成
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        elif fixed_seed is False:
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)

        return self.instrument_values




















