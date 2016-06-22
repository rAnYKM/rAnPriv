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

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_weight(set_a, res, set_s):
    if len(set_a & res) == 0:
        logging.error("knapsack: probably something goes wrong")
        return 0
    return len(set_a & set_s & res) / float(len(set_a & res))


def knapsack(items, max_weight, set_s, whole_set):
    """
    Solve the knapsack-like problem in ran_graph
    :param items: a sequence of tuple (value, w_set)
    :param max_weight: a non-negative integer
    :param set_s: the set of secret
    :param whole_set: the whole set
    :return : best value, and the corresponding index list.
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    def best_value(i, j, res):
        if i == 0:
            return 0
        value, w_set = items[i - 1]
        weight = get_weight(w_set, res, set_s)
        if weight > j:
            return best_value(i - 1, j, res)
        else:
            return max(best_value(i - 1, j, res),
                       best_value(i - 1, j, res & w_set) + value)

    j = max_weight
    result = []
    res = whole_set
    for i in xrange(len(items), 0, -1):
        if best_value(i, j, res) != best_value(i - 1, j, res):
            # result.append(items[i - 1])
            result.append(i - 1)
            res = res & items[i - 1][1]
            # print best_value(i - 1, j, res), result, res
            # j -= items[i - 1][1]
    result.reverse()
    return best_value(len(items), max_weight, whole_set), result
