# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ran_kp.py
# Date: Jun. 29, 2016


class Knapsack:
    """
    Knapsack problem solver
    .items
        (id, value -> p, weight -> w)
    .max_weight
        -> c

    >>> items = [(0, 4, 12), (1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)]
    >>> Knapsack(items, 15).dp_solver()
    (11, [(1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)])
    """

    def __init__(self, items, max_weight):
        self.items = items
        self.max_weight = max_weight
        self.sorted_items = sorted(self.items, key=lambda tup: float(tup[1])/tup[2], reverse=True)

    def dp_solver(self):
        def dp_recursive(i, j):
            if i == 0: return 0
            id, value, weight = self.items[i - 1]
            if weight > j:
                return dp_recursive(i - 1, j)
            else:
                return max(dp_recursive(i - 1, j),
                           dp_recursive(i - 1, j - weight) + value)

        j = self.max_weight
        n = len(self.items)
        result = []
        for i in xrange(n, 0, -1):
            if dp_recursive(i, j) != dp_recursive(i - 1, j):
                result.append(self.items[i - 1])
                j -= self.items[i - 1][1]
        result.reverse()
        return dp_recursive(n, self.max_weight), result

def test():
    items = [(0, 4, 12), (1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)]
    kp = Knapsack(items, 15)
    print kp.sorted_items
    print kp.dp_solver()

if __name__ == '__main__':
    test()
