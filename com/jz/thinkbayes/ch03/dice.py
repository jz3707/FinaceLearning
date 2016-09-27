#!/usr/bin/env python
# coding=utf-8

# dice question


from __future__ import print_function, division

from ..thinkbayes2 import Suite

class Dice(Suite):

    """
    Represents hypotheses about which die was rolled.
    """

    def Likelihood(self, data, hypo):
        """
        Computes the likelihood of the data under the hypothesis
        :param data: integer number of sides of the die
                骰子的面
        :param hypo: integer die roll
                骰子的点数
        :return:
        """

        if hypo < data:
            return 0
        else:
            return 1.0 / hypo


def main():
    suite = Dice([4, 6, 8, 12, 20])
    suite.Update(6)
    print('After one 6')
    suite.Print()


    for roll in [4, 8, 7, 7, 2]:
        suite.Update(roll)

    print('After more rolls')
    suite.Print()

if __name__ == '__main__':
    main()






