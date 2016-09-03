#!/usr/bin/env python
# coding=utf-8

# random generate number

import numpy as np


def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=False):
    """

    :param fixed_seed:
    :param shape:
    :param antithetic: 默认采用antithetic抽样
    :param moment_matching: 默认采用moment_matching抽样
    :return:
    """

    if fixed_seed:
        np.random.seed(1000)

    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] / 2))
        ran = np.c_[ran, -ran]

    else:
        ran = np.random.standard_normal(shape)

    if moment_matching:
        ran -= np.mean(ran)
        ran /= ran.std()

    if shape[0] == 1:
        return ran[0]
    else:
        return ran
