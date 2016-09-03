#!/usr/bin/env python
# coding=utf-8

import pandas
from collections import Counter
import copy
import math
import logging
import numpy as np
import random


DEFAULT_LABEL = '_nolegend_'

class _DictWrapper(object):
    """
    An object that contains a dictionary.
    """

    def __init__(self, obj=None, label=None):
        """
        initializes the distribution
        :param obj: Hist, Pmf, Pdf, dict, pandas, Series, list of pairs
        :param label: string label
        :return:
        """

        self.label = label if label is not None else DEFAULT_LABEL
        self.d = {}

        # flag whether the distribution is under a log transform
        self.log = False

        if obj is None:
            return

        # if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
        if isinstance(obj, _DictWrapper):
            self.label = label if label is not None else obj.label

        if isinstance(obj, dict):
            self.d.update(obj.items())
        # elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
        elif isinstance(obj, _DictWrapper):
            self.d.update(obj.Items())
        elif isinstance(obj, pandas.Series):
            self.d.update(obj.value_counts().iteritems())
        else:
            # finally, treat it like a list
            self.d.update(Counter(obj))


        if len(self) > 0 and isinstance(self. Pmf):
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
            return '%s(%s)' % (cls, str(self.d))
        else:
            return '%s(%s, %s)' % (cls, repr(self.d), repr(self.label))

    def __eq__(self, other):
        return self.d == other.d

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        """
        Return an iterator over keys.
        :return:
        """
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, item):
        return self.d.get(item)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]


    def Remove(self, x):
        """
        Remove a value
        Throws an exception uf the value is not there
        :param x:
        :return:
        """

        del self.d[x]

    def MaxLike(self):
        """
        Return the largest frequency / probability in the map.
        :return:
        """

        return max(self.d.values())


    def Set(self, x, y=0):
        """
        Sets the freq/prob associated with the value x
        :param x: number value
        :param y: number freq or prob
        :return:
        """
        self.d[x] = y

    def Incr(self, x, term=1):
        """
        Increments the freq/prob associated with the value x.
        :param x: number value
        :param term: how much to increment by
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

    def Total(self):
        """
        Return the total of the freq / prob in the map.
        :return:
        """

        total = sum(self.d.values())
        return total


    def Largest(self, n=10):
        """
        Return the largest n values, with freq/prob
        :param n: number of items to return
        :return:
        """
        return  sorted(self.d.items(), reverse=True)[:n]


    def Samllest(self, n=10):
        """
        Return the smallest n values, with freq/prob
        :param n: number of items to return
        :return:
        """
        return  sorted(self.d.items(), reverse=False)[:n]


    def Items(self):
        """
        Gets an unsorted sequence of (value, freq/prob) pairs.
        :return:
        """
        return self.d.items()

    def Render(self, **options):
        """
        Generates a sequence of points suitable of plotting.
        Note : options are ignored
        :param options:
        :return: tuple of (sorted value sequence, freq/prob sequence)
        """
        if min(self.d.keys()) is np.nan:
            logging.warning("Hist: contains Nan, may not render correctly.")

        return zip(*sorted(self.Items()))


    def MakeCdf(self, label=None):
        """
        Make a Cdf.
        :param label:
        :return:
        """

        label = label if label is not None else self.label
        return Cdf(self, label=label)

    def Print(self):
        """
        prints the values and freq/prob in ascending order.
        :return:
        """
        for val, prob in sorted(self.d.items()):
            print(val, prob)

    def Copy(self, label=None):
        """
        Return a copy.
        make a shadow of d.
        if want to deep copy of d, use copy.deepcopy on the whole object.

        :param label:s tring label for new Hist
        :return: new _DictWrapper with the samce type
        """

        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = copy.copy(label if label is not None else self.label)

        return new

    def Scale(self, factor):
        """
        Multiplies the values by a factor.
        :param factor: what to multiply by
        :return:  new object
        """

        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        """
        Log transform the probabilities.
        Removes values with probability 0.
        Normalizes so that the largest logprob is 0.
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
        Exponentiates  the probabilities.
        :param m: how much to shift the ps before exponentiating
        if m is None, normalizes so that the largest prob is 1.
        :return:
        """

        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform.")

        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))


    def GetDict(self):
        """
        Gets the dictionary.
        :return:
        """

        return self.d

    def SetDict(self, d):
        """
        Sets the dictionary.
        :param d:
        :return:
        """

        self.d = d

    def Values(self):
        """
        Gets an unsorted sequence of values.

        :return:
        """

        return self.d.keys()






class Pmf(_DictWrapper):
    """
    Represents a probability mass function
    表示概率函数。
    Values can be any hashcode type, probobilities are floating-point.
    Pmfs are not necessarily normalized
    """

    def Prob(self, x, default=0):
        """
        Gets the prob associate with the value x.
        :param x: number value
        :param default: value to return if the key is not there
        :return: float prob
        """

        return self.d.get(x, default)


    def Probs(self, xs):

        """
        Gets prob for a sequence of values.
        :param xs:
        :return:
        """
        return [self.Prob(x) for x in xs]


    def Percentile(self, percentage):
        """
        Computes a percentile of a given Pmf
        :param percentage: float 0 - 100
        :return: value from th Pmf
        """

        p = percentage / 100
        total = 0
        for val, prob in sorted(self.Items()):
            total += prob
            if total >= p:
                return val

    def ProbGreather(self, x):
        """
        Prob that sample from this Pmf exceeds x.
        :param x:  number value
        :return: float prob
        """
        if isinstance(x, _DictWrapper):
            return PmfProbGreather(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)

    def ProbLess(self, x):
        """
                Prob that sample from this Pmf exceeds x.
                :param x:  number value
                :return: float prob
                """
        if isinstance(x, _DictWrapper):
            return PmfProbLess(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)


    def __lt__(self, other):
        """
        Less than
        :param other: number or _DictWrapper
        :return:  float prob
        """

        return self.ProbLess(other)

    def __gt__(self, other):
        """
        Greather than
        :param other: number of _DictWrapper
        :return:  float prob
        """

        return self.ProbGreather(other)

    def __ge__(self, other):
        """
        Greather or equal
        :param other: number of _DictWrapper
        :return:  float prob
        """

        return 1 - (self < other)

    def __le__(self, other):
        """
        Less than or equal
        :param other: number of _DictWrapper
        :return:  float prob
        """

        return 1 - (self > other)

    def Normalize(self, fraction=1):
        """
        Normalizes this Pmf so the sum of all point si fraction.
        :param fraction: wht th total should be after normalization.
        :return: the total prob before normalizing.
        """

        if self.log:
            raise ValueError("Normalize : Pmf is under a log transform.")

        total = self.Total()
        if total == 0:
            raise ValueError("Normalize : total prob is 0.")

        factor = fraction / total
        for x in self.d:
            self.d[x] *= factor

        return total


    def Random(self):
        """
        Chooses a random element  from this Pmf.

        :return: float value from pmf.
        """

        target = random.random()
        total = 0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we should be get here
        raise ValueError("Random : Pmf might not be normalized.")


    def Mean(self):
        """
        Computes the mean of a Pmf
        :return: float mean
        """

        return sum(p * x for x, p in self.Items())












































