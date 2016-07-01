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
        def dp_recursive(p, q):
            if p == 0:
                return 0
            _, value, weight = self.items[p - 1]
            if weight > q:
                return dp_recursive(p - 1, q)
            else:
                return max(dp_recursive(p - 1, q),
                           dp_recursive(p - 1, q - weight) + value)

        j = self.max_weight
        n = len(self.items)
        result = []
        for i in xrange(n, 0, -1):
            if dp_recursive(i, j) != dp_recursive(i - 1, j):
                result.append(self.items[i - 1])
                j -= self.items[i - 1][2]
        result.reverse()
        return dp_recursive(n, self.max_weight), result

    def bnb_solver(self):
        class Node:
            def __init__(self, level, weight, value):
                self.level = level
                self.weight = weight
                self.value = value
                self.bound = 0
                self.sel = []

            def get_bound(self, items, max_weight):
                if self.weight >= max_weight:
                    return 0
                p_bound = self.value
                j = self.level + 1
                w = self.weight
                while j < len(items) and w + items[j][2] <= max_weight:
                    w += items[j][2]
                    p_bound += items[j][1]
                    j += 1
                if j < len(items):
                    p_bound += (max_weight - w)*items[j][1]/float(items[j][2])
                self.bound = p_bound

        # [level, weight, value]
        u = Node(-1, 0, 0)
        v = Node(0, 0, 0)
        queue = [u]
        max_value = 0
        max_sel = []
        while len(queue) != 0:
            u = queue.pop(0)
            if u.level == len(self.sorted_items) - 1:
                continue
            v.level = u.level + 1
            v.weight = u.weight + self.sorted_items[v.level][2]
            v.value = u.value + self.sorted_items[v.level][1]
            if v.weight <= self.max_weight and v.value > max_value:
                v.sel.append(self.sorted_items[v.level])
                max_value = v.value
                max_sel = v.sel
            v.get_bound(self.sorted_items, self.max_weight)
            if v.bound > max_value:
                queue.append(v)
        return max_value, max_sel


def test():
    # items = [(0, 4, 12), (1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)]
    items = [(0, 40, 2), (1, 50, 3.14), (2, 100, 1.98), (3, 95, 5), (4, 30, 3)]
    kp = Knapsack(items, 10)
    print kp.sorted_items
    print kp.dp_solver()
    print kp.bnb_solver()

if __name__ == '__main__':
    test()
