#!/usr/bin/env python
# coding=utf-8

# https://github.com/AllenDowney/ThinkBayes2/blob/master/code/thinkbayes2.py

# 这个package是干什么的
import bisect

import copy
import logging
import math
import random
import re

from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd

import scipy
from scipy import stats
from scipy import special
from scipy import ndimage

from io import open

ROOT2 = math.sqrt(2)


# 画Hist，Pmf，Cdf时，如果没有出现说明，需要覆盖DEFAULT_LABEL
DEFAULT_LABEL = '_nolegend_'


class _DictWrapper(object):
    """
    An object that contains a dict.
    """

    def __init__(self, obj=None, label=None):
        """
        initialize the distribution.
        :param obj: Hist, Pmf, Cdf, dict, pandas series, list of pairs
        :param label: string label
        :return:
        """

        self.label = label if label is not None else DEFAULT_LABEL
        self.d = {}

        # flag whether the distribution is under a log transform
        self.log = False

        if obj is None:
            return

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = label if label is not None else obj.label
        elif isinstance(obj, dict):
            self.d.update(obj.Items())
        elif isinstance(obj, pd.Series):
            self.d.update(obj.value_counts().iteritems())
        else:
            # finally， treat it like a list
            self.d.update(Counter(obj))

        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()


    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return '%s(%s)' % (cls, str(self.d))
        else:
            return self.label

    def __repr__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return '%s(%s)' % (cls, repr(self.d))
        else:
            return '%s(%s, %s)' % (cls, repr(self.d), repr(self.label))


    def __eq__(self, other):
        return self.d == other.d

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, value):
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def iterkeys(self):
        return iter(self.d)

    def Copy(self, label=None):
        """
        Return a copy.
        Make a shallow copy of d.
        to deep copy, using the deepcopy.
        :param label: string label for the new hist.
        :return: new _DictWrapper with the same type.
        """

        new = copy.copy(self)
        new.d = self.d
        new.label = label if label is not None else self.label

        return new


    def Scale(self, factor):
        """
        Multiplies the values by a factor.
        :param factor: wht to multiplies by
        :return: new obj

        """

        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)

        return new





    def GetDict(self):
        """
        Gets the dictionary
        :return:
        """

        return self.d

    def SetDict(self, d):
        """
        Sets the dictionary
        :param d:
        :return:
        """

        self.d = d

    def Values(self):
        """
        get the unsorted sequence of values.

        这里注意：
            可能迷惑的是，字典的keys是Hist/Pmf的值，而字典的values是频率或者概率
        :return:
        """

        return self.d.keys()

    def Items(self):
        """
        Gets an unsorted sequence of (value, freq/prob) pairs.
        :return:
        """

        return self.d.items()

    def Render(self, **options):
        """
        generate a sequence of points suitable of plotting.
        :param options:
        :return: tuple of (sorted value sequence, freq/prob sequence)
        """

        if min(self.d.keys()) is np.nan:
            logging.warning("Hist: coantains Nan, may bot render correcyly.")

        return zip(*sorted(self.Items()))




    def Set(self, x, y=0):
        """
        Sets the freq/prob associated with the value x.
        :param x: number value
        :param y: number freq or prob
        :return:
        """

        self.d[x] = y

    def Incr(self, x, term=1):
        """
        Increments the freq/prob associated the value x.
        :param x: number value
        :param term: how much to incrment by
        :return:
        """

        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        """
        Scales the freq/prob associated with the value x
        :param x: number value
        :param factor: how much to multiply by
        :return:
        """

        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        """
        Remove  a value
        :param x: value to remove
        :return:
        """

        del self.d[x]

    def Total(self):
        """
        Returns the total of the frequencies/probabilitiees in the map.
        :return:
        """

        total = sum(self.d.values())
        return total

    def MaxLike(self):
        """
        Return the largest frequency/probability in the map.
        :return:
        """

        return max(self.d.values())

    def Largest(self, n=10):
        """
        Returns the largest n values, with freq/prob
        :param n:
        :return:
        """

        return sorted(self.d.items(), reverse=True)[:n]

    def Smallest(self, n=10):

        return sorted(self.d.items(), reverse=False)[:n]


    def MakeCdf(self, label=None):
        """
        Make a Cdf
        :param label:
        :return:
        """

        label = label if label is not None else self.label

        return Cdf(self, label=label)

    def Print(self):
        try:
            for val, prob in sorted(self.d.items()):
                print(val, prob)
        except:
            for val, prob in self.d.items():
                print(val, prob)


    def Log(self, m=None):
        """
        Log transform the probabilities
        Remove values with probalities 0.
        Normalizes so that teh largest logprob is 0.
        :param m:
        :return:
        """

        if self.log:
            raise ValueError("Pmf/Hist already under a log transform.")

        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        """
        Exponentiates the prob
        :param m: how much to shift the ps before exponentiating.
        :return:
        """

        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform.")

        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))


class Hist(_DictWrapper):
    """
    Represents a histogram, which is a map from values to freq.
    values can be any hashable type.
    freq are integer counters.
    """

    def Freq(self, x):
        """
        Gets the freq associated with the value x.
        :param x: number value
        :return: int frequency
        """

        return self.d.get(x, 0)

    def Freqs(self, xs):
        """
        gets freq for a sequence of values.
        :param xs:
        :return:
        """

        return [self.Freq(x) for x in xs]


    def IsSubset(self, other):
        """
        Checks whether the values in this histogram are a subset of
        the values in the given histogram.
        :param other:
        :return:
        """
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True


    def Subtract(self, other):
        """
        Subtracts the values in the given histogram from this histogram.
        :param other:
        :return:
        """

        for val, freq in other.Items():
            self.Incr(val, -freq)



class Pmf(_DictWrapper):
    """
    Represents a probability mass function.
    概率module函数

    values可以是hashable类型。概率是浮点类型。
    Pmfs不必归一化。
    """

    def Prob(self, x, default=0):
        """
        Gets prob associate with the value x..
        :param x: number value
        :param default: value to return if the key is not there
        :return:
        """

        return self.d.get(x, default)


    def Probs(self, xs):
        """
        Gets probabilities for a sequence of values.
        :param xs:
        :return:
        """

        return [self.Prob(x) for x in xs]


    def Percentile(self, percentage):
        """
        computes a percentile of a given Pmf.

        计算分位数
        注意：这个不是非常有效。计算多个概率的时候，还是计算Cdf吧。

        :param percentage: 0-100 浮点数
        :return:
        """

        p = percentage / 100
        total = 0
        for val, prob in sorted(self.Items()):
            total += prob
            if total >= p:
                return val


    def ProbGreater(self, x):
        """
        prob that a sample from this Pmf exceeds x.
        :param x: number value
        :return:
        """

        if isinstance(x, _DictWrapper):
            return PmfProbGreater(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)



    def ProbLess(self, x):
        """
        Prob that a sample from this Pmf is less than x.
        :param x: number value
        :return:
        """

        if isinstance(x, _DictWrapper):
            return PmfProbLess(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)


    def __lt__(self, obj):
        """
        less than
        :param obj: number or _DictWrapper
        :return:
        """
        return self.ProbLess(obj)

    def __gt__(self, obj):
        """
        greater than
        :param obj: number of _DictWrapper
        :return:
        """

        return self.ProbGreater(obj)

    def __ge__(self, obj):
        """
        greater than or equal
        :param obj: number of _DictWrapper
        :return:
        """

        return 1 - (self < obj)

    def __le__(self, obj):
        """
        less than or equal
        :param obj: number or _DictWrapper
        :return:
        """

        return 1 - (self > obj)


    def Normalize(self, fraction=1):
        """
        Normalizes this PMF so the sum of all probs is fraction.
        :param fraction: wht the total should be after normalization.
        :return:
        """

        if self.log:
            raise ValueError("Normalize : Pmf is under a log transform.")

        total = self.Total()
        if total == 0:
            raise ValueError("Normalize : total probability is zero.")

        factor = fraction / total
        for x in self.d:
            self.d[x] *= factor

        return total


    def Random(self):
        """
        Choose a random element from this pmf.
        这不是很有效的方法，如果需要调用几次，考虑转换成CDF。

        随机的分位数
        :return:
        """

        target = random.random()
        total = 0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # 如果没获取到，就有问题了。
        raise ValueError("Random : Pmf might not be normalized.")


    def Mean(self):
        """
        compute the mean of a Pmf
        期望
        :return:
        """

        return sum(p * x for x, p in self.Items())


    def Var(self, mu=None):
        """
        compute the variance of a pmf.
        :param mu: the point around which the variance is computed
            if omitted, computes the mean.
        :return:
        """
        if mu is None:
            mu = self.Mean()

        return sum(p * (x - mu) ** 2 for x, p in self.Items())


    def Expect(self, func):
        """
        computes the expectation of func(x)
        :param func:
        :return:
        """

        return np.sum(p * func(x) for x, p in self.Items())


    def Std(self, mu=None):
        """
        std
        :param mu:
        :return:
        """

        var = self.Var()
        return math.sqrt(var)


    def MaximumLikelihood(self):
        """
        Returns the value with the highest probability.
        :return:
        """

        _, val = max((prob, val) for val, prob in self.Items())
        return val


    def CredibleInterval(self, percentage=90):
        """
        Compute the central credible interval.
        计算中心的置信区间。
        如果percentage=90，计算90CI
        :param percentage:0-100 float
        :return: 返回两个float的序列，一个是low一个是high

        """

        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)


    def AddPmf(self, other):
        """
        computes the pmf of the sum of values drawn from self and other.
        :param other:
        :return:
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 + v2, p1 * p2)

        return pmf

    def AddConstant(self, other):
        """
        Computes the Pmf of the sum a constant and values from self.
        :param other: a number value
        :return:
        """

        if other == 0:
            return self.Copy()

        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 + other, p1)

        return pmf

    def __add__(self, other):
        """
        computes the pmf of the sum of values drawn from self and other.
        :param other: pmf or a scaler
        :return: new Pmf
        """

        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)


    def SubPmf(self, other):
        """
        computes the pmf of diff of values drawn from self and other.
        :param other:  pmf
        :return:
        """

        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)

        return pmf

    def __sub__(self, other):
        """
        computes the pmf of the diff of values drawn from self and other
        :param other:
        :return:
        """

        try:
            return self.SubPmf(other)
        except AttributeError:
            return self.AddConstant(-other)


    def MulPmf(self, other):
        """
        computes the pmf of the diff of values drawn from self and other
        :param other:  pmf
        :return:
        """

        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 * v2, p1 * p2)

        return pmf


    def MulConstant(self, other):
        """
        computes the pmf of the diff of values drawn from self and other
        :param other:  a number value
        :return:
        """

        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 * other, p1)

        return pmf


    def __mul__(self, other):
        """
        computes the pmf of the diff of values drawn from self and other
        :param other:  pmf
        :return:
        """

        try:
            return self.MulPmf(other)
        except AttributeError:
            return self.MulConstant(other)


    def DivPmf(self, other):
        """
        computes the pmf of the ratio of values drawn from self and other
        :param other: pmf
        :return:
        """

        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 / v2, p1 * p2)

        return pmf


    def __div__(self, other):
        """
        computes the pmf of the ratio of values drawn from self and other
        :param other: pmf
        :return:
        """

        try:
            return self.DivPmf(other)
        except AttributeError:
            return self.MulConstant(1 / other)


    def Max(self, k):
        """
        computes the cdf of the maximum of k selections from this dist.
        :param k: top k
        :return: new Cdf
        """

        cdf = self.MakeCdf()
        return cdf.Max(k)



class Joint(Pmf):
    """
    Represents a joint distribution.
    the value are sequences(usually tuples)
    """

    def Marginal(self, i, label=None):
        """
        gets the marginal distribution of the indicated variable.
        计算边缘概率
        :param i: index of the variable we want
        :param label:
        :return: pmf
        """

        pmf = Pmf(label=label)
        for vs, prob in self.Items():
            pmf.Incr(vs[i], prob)

        return pmf


    def Conditional(self, i, j, val, label=None):
        """
        gets the conditional distribution of the indicated variable.
        distribution of vs[i], conditioned on vs[j] = val

        计算条件概率。
        P(vs[j]) = val
        P(vs[i]|vs[j])
        :param i: index of the variable we want
        :param j: which variable is conditioned on
        :param val: the value the jth variable has to have
        :param label:
        :return: pmf
        """

        pmf = Pmf(label=label)

        for vs, prob in self.Items():
            if vs[j] != val:
                continue
            pmf.Incr(vs[i], prob)

        pmf.Normalize()
        return pmf


    def MaxLikeInterval(self, percentage=90):
        """
        return the maximum-likelihood credible interval.
        返回最大似然置信区间

        :param percentage: 0 - 100 float
        :return: list of values from the suite
        """

        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.Items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob

            if total >= percentage / 100:
                break

        return interval



def MakeJoint(pmf1, pmf2):
    """
    joint distribution of values from pmf1 and pmf2.
    联合分布
    假设pmf表示独立随机变量。
    :param pmf1:
    :param pmf2:
    :return:
    """

    joint = Joint()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            joint.Set((v1, v2), p1 * p2)

    return joint


def MakeHistFromList(t, label=None):
    """
    Make a histogram from an unsorted sequence of values.
    :param t: sequence of numbers
    :param label: string label for the histogram
    :return:
    """

    return Hist(t, label=label)


def MakeHistFromDict(d, label=None):
    """
    Make a histogram from a map from values to freq
    :param d: dictionary thar maps values to freq
    :param label: string label for the histogram
    :return:
    """

    return Hist(d, label)


def MakePmfFromList(t, label=None):
    """
    Make a Pmf from an unsorted sequence of values.
    :param t: sequence of numbers
    :param label: string label for the Pmf
    :return:
    """

    return Pmf(t, label=label)

def MakePmfFromDict(d, label=None):
    """
    Make a Pmf from a map from values to freq
    :param d: dictionary thar maps values to freq
    :param label: string label for the Pmf
    :return:
    """

    return Pmf(d, label=label)


def MakePmfFromItems(t, label=None):
    """
    Make a Pmf from an unsorted sequence of value-prob pairs.
    :param t: sequence of value-prov pairs
    :param label: string label for the Pmf
    :return:
    """

    return Pmf(dict(t), label=label)

def MakePmfFromHist(hist, label=None):
    """
    Make a normalized Pmf from a hist obj
    :param hist:
    :param label:
    :return:
    """

    if label is None:
        label = hist.label

    return Pmf(hist, label=label)

def MakeMixture(metapmf, label='mix'):
    """
    make a mixture distribution.
    :param metapmf: pmf that maps from pmfs to probs
    :param label:
    :return:
    """
    mix = Pmf(label=label)
    for pmf, p1 in metapmf.Items():
        for x, p2 in pmf.Items():
            mix.Incr(x, p1 * p2)

    return mix

def MakeUniformPmf(low, high, n):
    """
    make a uniform pmf
    :param low: lowest value (包括在内)
    :param high: highest value (包括在内)
    :param n: numebr of values
    :return:
    """
    pmf = Pmf()
    for x in np.linspace(low, high, n):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf


class Cdf:
    """
    represents a cumulative distribution function.

    xs: sequence of values
    ps: sequence of prob
    label : string used as a graph label.
    """

    def __init__(self, obj=None, ps=None, label=None):
        """
        if ps is provided, obj must bu the corresponding list of values.

        :param obj: hist, pmf, cdf, pdf, dict, pandas series, list of pairs.
        :param ps: list of cumulative prob
        :param label: string label
        :return:
        """

        self.label = label if label is not None else DEFAULT_LABEL

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            if not label:
                self.label = label if label is not None else obj.label

        if obj is None:
            # caller does not provide obj, make an empty Cdf.
            self.xs = np.asarray([])
            self.ps = np.asarray([])

            if ps is not None:
                logging.warning("Cdf : can't pass ps without also passing xs.")
            return

        else:
            # if the caller provides xs and ps, just store them.
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Cdf : ps can't be a string")

                self.xs = np.asarray(obj)
                self.ps = np.asarray(ps)
                return

        # caller has provided just obj, not ps
        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)

            return


        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)

        if len(dw) == 0:
            self.xs = np.asarray([])
            self.ps = np.asarray([])

            return

        xs, freqs = zip(*sorted(dw.Items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=np.float)
        # 归一化
        self.ps /= self.ps[-1]


    def __str__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return '%s(%s, %s)' % (cls, str(self.xs), str(self.ps))
        else:
            return '%s(%s, %s, %s)' %(cls, str(self.xs), str(self.ps), str(self.label))


    def __len__(self):
        return len(self.xs)

    def __getitem__(self, x):
        # 这里有点问题
        return self.Prob(x)

    def __setitem__(self, key, value):
        raise UnimplementedMethodException()

    def __delitem__(self, key):
        raise UnimplementedMethodException()

    def __eq__(self, other):
        return np.all(self.xs == other.xs) and np.all(self.ps == other.ps)

    def Print(self):
        for val, prob in zip(self.xs, self.ps):
            print(val, prob)

    def Copy(self, label=None):

        if label is None:
            label = self.label

        return Cdf(list(self.xs), list(self.ps), label=label)

    def MakePmf(self, label=None):

        if label is None:
            label = self.label

        return Pmf(self, label=label)

    def Items(self):
        """
        return a sorted sequence of (value, prov) pairs
        注意：py3中返回一个iterator
        :return:
        """

        a = self.ps
        b = np.roll(a, 1)
        b[0] = 0

        return zip(self.xs, a - b)


    def Shift(self, term):
        """
        add a term to xs
        :param term:
        :return:
        """

        new = self.Copy()
        # don't use +=
        new.xs = new.xs + term
        return new


    def Scale(self, factor):
        """
        multiply the xs by the factor
        :param factor:
        :return:
        """

        new = self.Copy()
        new.xs = new.xs * factor

        return new


    def Prob(self, x):
        """
        Return CDF(x), the prob that corresponds to value x.
        :param x: number value
        :return:
        """

        if x < self.xs[0]:
            return 0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]

        return p


    def Probs(self, xs):
        """
        get prob for a sequence of values.
        :param xs: any sequence that can be converted to numpy array
        :return:
        """

        xs = np.asarray(xs)
        index = np.searchsorted(self.xs, xs, side='right')
        ps = self.ps[index + 1]
        ps[xs < self.xs[0]] = 0

        return ps

    ProbArray = Probs()


    def Value(self, p):
        """
        return inverseCDF(p), the value that corresponds to prob p
        :param p:
        :return:
        """

        if p < 0 or p > 1:
            raise ValueError("Probability p must by in range [0, 1]")

        # 排序插入，可插入重复值
        index = bisect.bisect(self.ps, p)
        return self.xs[index]


    def Values(self, ps=None):
        """
        returns inverseCDF(p),, the value that corresponds to prob p.
        if ps is not provided, returns all values.
        :param ps:
        :return:
        """

        if ps is None:
            return self.xs

        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError("Probability p must by in range [0, 1]")

        # 保持原有顺序的插入，并返回插入index
        index = np.searchsorted(self.ps, ps, side='left')

        return self.xs[index]

    ValueArray = Values()


    def Percentile(self, p):
        """
        return the value that corresponds to percentile p.
        返回p对应百分位的x值
        :param p:
        :return:
        """

        return self.Values(p / 100)


    def Percentiles(self, ps):
        """
        return the value that corresponds to percentile ps.
        :param ps:
        :return:
        """

        ps = np.asarray(ps)
        return self.Values(ps)


    def PercentileRank(self, x):
        """
        Return the percentile rank of the value x.
        :param x: potential value in cdf, cdf中的潜在的值
        :return:
        """

        return self.Prob(x) * 100


    def PercentileRanks(self, xs):
        """
        Return the percentile rank of the value xs.
        :param xs: potential value in cdf, cdf中的潜在的值
        :return:
        """

        return self.Probs(xs) * 100


    def Random(self):
        """
        chooses a random value from this distribution.
        :return:
        """

        return self.Value(random.random())


    def Sample(self, n):
        """
        generates a random sample from this distribution.
        :param n:
        :return:
        """

        ps = np.random.random(n)

        return self.ValueArray(ps)


    def Mean(self):
        """
        computes the mean of cdf
        这个有问题，为什么减去old啊，怎么搞的？？？

        :return:
        """

        old_p = 0.
        total = 0.
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p

        return total


    def CredibleInterval(self, percentage=90):
        """
        Computes the central credible interval.
        :param percentage: 90
        :return:
        """

        prob = (1 -  percentage / 100) / 2
        interval = self.Value(prob), self.Value(1 - prob)

        return interval


    def _Round(self, multiplier=1000):
        """
        An entry is added to the cdf only if the percentile differs from
        the previous value in a significant digit, where the number of signficant digits
        is determined by multiplier.
        The default is 1000, which keeps log10(1000) = 3.
        :param multiplier:
        :return:
        """

        raise UnimplementedMethodException()


    def Render(self, **options):
        """
        generates a sequence of points, suitable fro plotting.

        :param options:
        :return:
        """

        def interleave(a, b):
            c = np.empty(a.shape[0] + b.shape[0])
            c[::2] = a
            c[1::2] = b
            return c

        a = np.array(self.xs)
        xs = interleave(a, a)

        shift_ps = np.roll(self.ps, 1)
        shift_ps[0] = 0

        ps = interleave(shift_ps, self.ps)

        return  xs, ps


    def Max(self, k):
        """
        computes the cdf of the maximim of k selections from this dist.
        :param k: int
        :return:
        """

        cdf = self.Copy()
        cdf.ps **= k
        return cdf


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""



def MakeCdfFromItems(items, label=None):
    """
    make a cdf from an unsorted sequence of (value, frequency) pairs.
    :param items: unsorted sequence of (value, freq) pairs
    :param label: string label for this CDF
    :return:
    """

    return Cdf(dict(items), label=label)


def MakeCdfFromDict(d, label=None):
    """
    make cdf from a dict.
    :param d:
    :param label:
    :return:
    """

    return Cdf(d, label=label)


def MakeCdfFromList(seq, label=None):
    """
    create cdf from an unsorted sequence.
    :param seq:
    :param label:
    :return:
    """

    return Cdf(seq, label=label)


def MakeCdfFromHist(hist, label=None):
    """
    make cdf from a hist obj
    :param hist:
    :param label:
    :return:
    """

    if label is None:
        label = hist.label

    return Cdf(hist, label=label)


def MakeCdfFromPmf(pmf, label=None):
    """
    make a cdf from a pmf obj
    :param pmf:
    :param lable:
    :return:
    """

    if label is None:
        label = pmf.label

    return Cdf(pmf, label=label)



class Suite(Pmf):
    """
    represents a suite of hypotheses and their prob.
    """

    def Likelihood(self, data, hypo):
        """
        computes the likelihood of data under the hypothesis.
        :param data: some representation of the hypothesis
        :param hypo: some representation of the data
        :return:
        """

        raise UnimplementedMethodException()


    def LogLikelihood(self, data, hypo):
        """
        computes the likelihood of data under the hypothesis.
        :param data: some representation of the hypothesis
        :param hypo: some representation of the data
        :return:
        """

        raise UnimplementedMethodException()


    def Update(self, data):
        """
        updates each hypothesis based on the data.
        :param data: any representation of the data
        :return:
        """

        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Incr(hypo, like)


    def UpdateSet(self, dataset):
        """
        updates each hypothesis based on the dataset.
        this is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.


        :param dataset:
        :return:
        """

        for data in dataset:
            for hypo in self.Values():
                like = self.Likelihood(data, hypo)
                self.Mult(hypo, like)

        return self.Normalize()

    def LogUpdateSet(self, dataset):
        """

        :param dataset:
        :return:
        """

        for data in dataset:
            self.LogUpdateSet(data)


    def Print(self):
        for hypo, prob in sorted(self.Incr()):
            print(hypo, prob)


    def MakeOdds(self):
        """
        Transform from prob to odds.
        prob = 0 removed
        :return:
        """

        for hypo, prob in self.Items():
            if prob:
                self.Set(hypo, Odds(prob))
            else:
                self.Remove(hypo)

    def MakeProbs(self):
        """
        transform from odds to prob
        :return:
        """

        for hypo, odds in self.Items():
            self.Set(hypo, Probability(odds))


def MakeSuiteFromHist(hist, label=None):
    """
    make a normalized suite from a hist obj
    :param hish:
    :param label:
    :return:
    """

    if label is None:
        label = hist.label

    d = dict(hist.GetDict())
    return MakeCdfFromDict(d, label=label)


def MakeSuiteFromDict(d, label=None):
    """

    :param d:
    :param label:
    :return:
    """

    suite = Suite(label=label)
    suite.SetDict(d)
    suite.Normalize()
    return suite


def MakeSuiteFromList(t, label=None):
    """
    make a suite from an unsorted sequence of values.
    :param t: sequence of numbers
    :param label: string label
    :return:
    """

    hist = MakeHistFromList(t, label=label)
    d = hist.GetDict()

    return MakeSuiteFromDict(d)



class Pdf(object):
    """
    represents a prob density function pdf.
    """


    def Density(self, x):
        """
        Evaluates this pdf at x.
        :param x:
        :return: float or numpy of prob density
        """

        raise UnimplementedMethodException()


    def GetLinspace(self):
        """
        get a linsapce for plotting.
        :return:
        """

        raise UnimplementedMethodException()

    def MakePmf(self, **options):
        """
        Make a discrete version of this pdf.
        离散形式的pdf。

        options 包括下面几种形式：
        label : string
        low : low end of range
        high : high end of range
        n: number of places to evaluate
        :param options:
        :return: pmf
        """

        label = options.pop('label', '-')
        xs, ds = self.Render(**options)

        return Pmf(dict(zip(xs, ds)), label=label)
























































































