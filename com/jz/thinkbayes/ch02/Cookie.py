#!/usr/bin/env python
# coding=utf-8


# cookie question

from .. thinkbayes2 import Pmf

class Cookie(Pmf):

    def __init__(self, hypos):

        """

        :type hypos: list
        """
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)

        self.Normalize()


    def Update(self, data):
        """
        Bayes formula:
        p(A|B) = p(B|A)p(A) / p(B)

        通过先验来更新后验
        like是似然，p(B|A)
        hypo是先验，p(A)
        :param data:
        :return:
        """

        # self.Values获取hypos
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)

        self.Normalize()


    # 后验，======!!太low!!=====
    mixes = {
        'Bowl1': dict(vanilla=0.75, chocolate=0.25),
        'Bowl2': dict(vanilla=0.5, chocolate=0.5),
    }


    def Likelihood(self, data, hypo):

        mix = self.mixes[hypo]
        like = mix[data]
        return like


def main():

    hypos = ['Bowl1', 'Bowl2']
    pmf = Cookie(hypos)

    pmf.Update('vanilla')

    for hypo, prob in pmf.Items():
        print(hypo, prob)


if __name__ == '__main__':
    main()



