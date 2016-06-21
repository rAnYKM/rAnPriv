# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ran_knapsack.py
# Date: Jun. 20, 2016
import collections
import functools


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)

def get_weight(set_a, res, set_s):
    if len(set_a & res) == 0:
        print "probably something goes wrong", set_a, res
        return 0
    return len(set_a & set_s & res) / float(len(set_a & res))


def knapsack(items, maxweight, set_s, whole_set):
    """
    Solve the knapsack-like problem in ran_graph
    :param items: a sequence of tuple (value, w_set)
    :param max_weight: a non-negative integer
    :return : best value, and the corresponding index list.

    >>> a = [1, 2, 3, 5]
    >>> b = [1, 2, 4]
    >>> c = [1, 3, 4, 5]
    >>> items = [(1, set(a)), (1, set(b)), (1, set(c))]
    >>> s = [1, 2]
    >>> print knapsack(items, 0.3, set(s))
    (2, [0, 1, 2])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    def bestvalue(i, j, res):
        if i == 0: return 0
        value, w_set = items[i - 1]
        weight = get_weight(w_set, res, set_s)
        if weight > j:
            return bestvalue(i - 1, j, res)
        else:
            return max(bestvalue(i - 1, j, res),
                       bestvalue(i - 1, j, res & w_set) + value)

    j = maxweight
    result = []
    res = whole_set
    for i in xrange(len(items), 0, -1):
        if bestvalue(i, j, res) != bestvalue(i - 1, j, res):
            # result.append(items[i - 1])
            result.append(i - 1)
            res = res & items[i - 1][1]
            # print bestvalue(i - 1, j, res), result, res
            # j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight, whole_set), result

if __name__ == '__main__':
    a = [1, 2, 3, 5]
    b = [1, 2, 4]
    c = [3, 4, 5]
    items = [(1, set(a)), (1, set(b)), (1, set(c))]
    s = [1, 2]