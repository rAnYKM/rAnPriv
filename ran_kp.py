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
                if self.weight > max_weight:
                    self.bound = 0
                    return
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
            v.sel = list(u.sel)
            if v.weight <= self.max_weight and v.value > max_value:
                v.sel.append(self.sorted_items[v.level])
                max_value = v.value
                max_sel = v.sel
            v.get_bound(self.sorted_items, self.max_weight)
            if v.bound > max_value:
                queue.append(v)
            nv = Node(v.level, u.weight, u.value)
            nv.get_bound(self.sorted_items, self.max_weight)
            nv.sel = list(u.sel)
            if nv.bound > max_value:
                queue.append(nv)
        return max_value, max_sel


class MultiDimensionalKnapsack:
    def __init__(self, items, max_weights):
        self.items = items
        self.max_weights = max_weights
        self.sorted_items = sorted(self.items, key=lambda tup: float(tup[1]) / sum(tup[2]), reverse=True)

    def dp_solver(self):
        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def reduce_weights(w, max_w):
            return [max_w[i] - w[i] for i in xrange(len(w))]

        def dp_recursive(p, q):
            if p == 0:
                return 0
            _, value, weights = self.items[p - 1]
            if exceed_weights(weights, q):
                return dp_recursive(p - 1, q)
            else:
                return max(dp_recursive(p - 1, q),
                           dp_recursive(p - 1, reduce_weights(weights, q)) + value)

        j = self.max_weights
        n = len(self.items)
        result = []
        for i in xrange(n, 0, -1):
            if dp_recursive(i, j) != dp_recursive(i - 1, j):
                result.append(self.items[i - 1])
                j = reduce_weights(self.items[i - 1][2], j)
        result.reverse()
        return dp_recursive(n, self.max_weights), result

    def bnb_solver(self):
        # TODO: There is something wrong with upper bound decision
        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def reduce_weights(w, max_w):
            return [max_w[i] - w[i] for i in xrange(len(w))]

        def increase_weights(w, max_w):
            return [max_w[i] + w[i] for i in xrange(len(w))]

        class Node:
            def __init__(self, level, weight, value):
                self.level = level
                self.weight = list(weight)
                self.value = value
                self.bound = 0
                self.sel = []

            def get_bound(self, items, max_weights):
                if exceed_weights(self.weight, max_weights):
                    self.bound = 0
                    return
                p_bound = self.value
                j = self.level + 1
                w = self.weight
                while j < len(items) and not exceed_weights(w, reduce_weights(items[j][2], max_weights)):
                    w = increase_weights(w, items[j][2])
                    p_bound += items[j][1]
                    j += 1
                if j < len(items):
                    p_bound += sum(reduce_weights(w, max_weights)) * items[j][1] / float(sum(items[j][2]))
                self.bound = p_bound

        # [level, weight, value]
        u = Node(-1, [0]*len(self.max_weights), 0)
        v = Node(0, [0]*len(self.max_weights), 0)
        queue = [u]
        max_value = 0
        max_sel = []
        while len(queue) != 0:
            u = queue.pop(0)
            if u.level == len(self.sorted_items) - 1:
                continue
            v.level = u.level + 1
            v.weight = increase_weights(u.weight, self.sorted_items[v.level][2])
            v.value = u.value + self.sorted_items[v.level][1]
            v.sel = list(u.sel)
            if not exceed_weights(v.weight, self.max_weights) and v.value > max_value:
                v.sel.append(self.sorted_items[v.level])
                max_value = v.value
                max_sel = v.sel
            v.get_bound(self.sorted_items, self.max_weights)
            if v.bound > max_value:
                queue.append(v)
            nv = Node(v.level, u.weight, u.value)
            nv.get_bound(self.sorted_items, self.max_weights)
            nv.sel = list(u.sel)
            if nv.bound > max_value:
                queue.append(nv)
        return max_value, max_sel


def test():
    # items = [(0, 4, 12), (1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)]
    items = [(0, 40, 2), (1, 50, 3.14), (2, 100, 1.98), (3, 95, 5), (4, 30, 3)]
    kp = Knapsack(items, 10)
    print kp.sorted_items
    print kp.dp_solver()
    print kp.bnb_solver()

    m_items = [(0, 5, (1, 2, 3)), (1, 3, (3, 1, 1)), (2, 8, (5, 1, 1)), (3, 2, (2, 2, 2)),
               (4, 4, (1, 5, 1)), (5, 10, (6, 2, 3))]
    m_weights = [8, 8, 8]
    mkp = MultiDimensionalKnapsack(m_items, m_weights)
    print mkp.sorted_items
    print mkp.dp_solver()
    print mkp.bnb_solver()

if __name__ == '__main__':
    test()
