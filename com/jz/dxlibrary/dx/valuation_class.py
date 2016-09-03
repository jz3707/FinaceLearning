#!/usr/bin/env python
# coding=utf-8

# valuation base class


class valuation_class(object):
    """
    这个类是单一因素的估值基类
    因为只考虑一种标的，不考虑标的之间的相关性。
    """

    def __init__(self, name, underlying, mar_env, payoff_func):

        try:

            self.name = name
            self.pricing_date = mar_env.pricing_date

            # 如果有执行价格就传进来，要是没有就略过
            try:
                self.stride = mar_env.get_constant('stride')
            except:
                pass

            self.maturity = mar_env.get_constant('maturity')
            self.currency = mar_env.get_constant('currency')

            # 模拟
            self.frequency = underlying.frequency
            self.paths = underlying.paths
            self.discount_curve = underlying.discount_curve
            self.payoff_func = payoff_func
            self.underlying = underlying
            self.underlying.special_dates.extends([self.pricing_date, self.maturity])

        except:
            print("Error parsing market environment.")



    def update(self, initial_value=None, volatility=None, stride=None, maturity=None):

        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)

        if volatility is not None:
            self.underlying.update(volatility=volatility)

        if stride is not None:
            self.stride = stride

        if maturity is not None:
            self.maturity = maturity

            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)

        self.underlying.instrument_values = None


    def delta(self, interval=None, accuracy=4):
        # 标的资产变动时候 期权的变动情况
        # 每一个标的资产的变化间距，标的资产是有变化的，那么这个变化的量就是interval
        interval = self.underlying.initial_value / 50.
        # present_value就是计算期权价值
        value_left = self.present_value(fixed_seed=True)
        initial_del = self.underlying.initial_value + interval

        self.underlying.update(initial_value=initial_del)
        value_right = self.present_value(fixed_seed=True)

        self.underlying.update(initial_value=initial_del - interval)

        # 期权的价值的变动， 范围 -1， +1
        delta = (value_right - value_left) / interval

        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)


    def vega(self, interval=0.01, accuracy=4):
        # 如果 interval 比 波动率的50分之一还小的话就将interval修改成波动率的50分之一
        if interval < self.underlying.volatility < 50.:
            interval = self.underlying.volatility < 50.

        value_left = self.present_value(fixed_seed=True)

        vola_del = self.underlying.volatility + interval
        self.underlying.update(volatility=vola_del)
        value_right = self.present_value(fixed_seed=True)

        self.underlying.update(volatility=vola_del - interval)

        vega = (value_right - value_left) / interval

        return round(vega, accuracy)
