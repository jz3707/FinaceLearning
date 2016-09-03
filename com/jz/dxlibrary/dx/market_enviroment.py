#!/usr/bin/env python
# coding=utf-8

# market environment

class market_enviroment(object):
    """
    初始化用于估值的市场环境相关信息
    """

    def __init__(self, name, pricing_date):
        """

        :param name:
        :param pricing_date: 定价基准日期
        :return:
        """

        self.name = name
        self.pricing_date = pricing_date

        # 常量，用于保存模型的参数或者期权到期日期
        self.constants = {}
        # 建模的对象
        self.lists = {}
        # 贴现因子类的实例
        self.curves = {}


    def add_constant(self, key, constant):
        """
        给constants添加元素
        :param key:
        :param constant:
        :return:
        """

        self.constants[key] = constant

    def get_constant(self, key):
        """
        get the constant value
        :param key:
        :return:
        """

        constant = self.constants.get(key)

        if constant is None:
            raise ValueError("the Key : %s is not in market environment constants!")

        return constant

    def add_list(self, key, list_object):
        """
        add the list
        :param key:
        :param list_object:
        :return:
        """

        self.lists[key] = list_object

    def get_list(self, key):
        """
        get the list
        :param key:
        :return:
        """
        list = self.lists.get(key)

        if list is None:
            raise ValueError("the Key : %s is not in market environment lists!")

        return list

    def add_curve(self, key, curve):
        """
        add curve
        :param key:
        :param curve:
        :return:
        """

        self.curves[key] = curve

    def get_curve(self, key):
        """
        get the curve
        :param key:
        :return:
        """

        curve = self.curves.get(key)

        if curve is None:
            raise ValueError("the Key : %s is not in market environment curve!")

        return curve


    def add_environment(self, env):
        """
        add the constant, list, curve
        如果某这个值存在，这里进行覆盖
        :param env:
        :return:
        """

        for key in env.constants:
            self.add_constant(key, env.constants[key])

        for key in env.lists:
            self.add_list(key, env.lists[key])

        for key in env.curves:
            self.add_curve(key, env.curves[key])


