#!/usr/bin/env python
# coding = utf-8

# 看涨期权定价BSN

def cal_bsm_value(S0, K, T, r, sigma):
    """
    C(St, K, t, T, r, sigma) = StN(d1) - e^(-r(T-t))KN(d2)
    d1 = (log(St/k) + (r+sigma**2/2)*(T-t)) / sigma*sqrt(T-t)
    d2 = (log(St/k) + (r-sigma**2/2)*(T-t)) / sigma*sqrt(T-t)
    N 是标准正态分布
    :param S0: 0时刻标的资产
    :param K: 期权的执行价格
    :param T: 期权的到期价格
    :param r: 无风险短期利率
    :param sigma: 期权的波动率
    :return:
    """

    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    # 从0时刻开始，所以T-t=T
    d1 = (log(S0 / K) + (r + sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - sigma ** 2) * T) / (sigma * sqrt(T))

    # cdf : cumulative distribution function, 累计分布函数, μ=0.0，σ=1.0，即标准正太分布
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)) - K * exp(-r*T) * stats.norm.cdf(d2, 0.0, 1.0)

    return value

def cal_bsm_vega(S0, K, T, r, sigma):
    """
    vega实际上是隐含波动率变动的时候，期权的价格变动的情况。
    因为使用牛顿法来解，所以使用了导数。
    隐含波动率的迭代公式如下：
    simga_imp_n+1 = simga_imp_n - (C(simga_imp_n) - C0) / (vega(sigma_imp_n))
    vega = St * N'(d1) * sqrt(T - t)
        t这里等于0

    N 是标准正态分布， N'是N的密度函数。
    :param S0: 0时刻标的资产
    :param K: 期权的执行价格
    :param T: 期权的到期价格
    :param r: 无风险短期利率
    :param sigma: 期权的波动率
    :return: vega value
    """
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    # 从0时刻开始，所以T-t=T
    d1 = (log(S0 / K) + (r + sigma ** 2) * T) / (sigma * sqrt(T))

    # cdf : probability distribution function, 密度分布函数, μ=0.0，σ=1.0，即标准正太分布
    value = (S0 * stats.norm.pdf(d1, 0.0, 1.0)) * sqrt(T)

    return value


def cal_bsm_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    """
    通过牛顿法和vage的方式来求隐含波动率
    隐含波动率的迭代公式如下：
    simga_imp_n+1 = simga_imp_n - (C(simga_imp_n) - C0) / (vega(sigma_imp_n))
    vega = St * N'(d1) * sqrt(T - t)
        t这里等于0

    :param S0: 0时刻标的资产
    :param K: 期权的执行价格
    :param T: 期权的到期价格
    :param r: 无风险短期利率
    :param C0: 0时刻期权的价格， 也就是市场价格
    :param sigma_est: 初始sigma值
    :param it: 迭代次数
    :return:
    """

    for i in range(it):
        # 通过S0,K，T, r, sigma_est计算出看涨期权的价格，然后再减去 - C0(市场价格)
        sigma_est -= ((cal_bsm_value(S0, K, T, r, sigma_est)) - C0) \
                / cal_bsm_vega(S0, K, T, r, sigma_est)

    return sigma_est




