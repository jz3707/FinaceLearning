#!/usr/bin/env python
# coding=utf-8

from .. thinkbayes2 import Pmf

pmf = Pmf()
# 先验
pmf.Set('Bowl1', 0.5)
pmf.Set('Bowl2', 0.5)

# P(C|B)
pmf.Mult('Bowl1', 0.75)
pmf.Mult('Bowl2', 0.5)

#
