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
        @memoized
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

    def greedy_solver(self, metrics='direct'):
        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def increase_weights(w, max_w):
            return [max_w[i] + w[i] for i in xrange(len(w))]

        if metrics == 'direct':
            ratio = lambda p, w, c: p/float(sum(w))
        elif metrics == 'scale':
            ratio = lambda p, w, c: p/float(sum([j/float(c[i]) for i, j in enumerate(w)]))
        else:
            print "ERROR: no such metrics"
            return
        greedy_items = sorted(self.items, key=lambda tup: ratio(tup[1], tup[2], self.max_weights), reverse=True)
        weight = [0]*len(self.max_weights)
        value = 0
        sel = []
        for item in greedy_items:
            if exceed_weights(increase_weights(weight, item[2]), self.max_weights):
                continue
            else:
                weight = increase_weights(weight, item[2])
                value += item[1]
                sel.append(item)
        return value, sel


class SetKnapsack:
    """
    Knapsack-like problem with non-linear constraints form
    Pr(S|A(x)) <= max_weight
    """
    def __init__(self, u_set, s_sets, items, max_weights):
        """
        initialize
        items ->
            id, value, set
        :param u_set: the universal set
        :param s_sets: a list of sets
        :param items: a list of tuple
        :param max_weights: a list of float
        """
        self.u_set = u_set
        self.s_sets = s_sets
        self.items = items
        self.max_weights = max_weights

    def dp_solver(self):
        def get_weight(set_a, s_sets):
            res_li = list()
            for set_s in s_sets:
                if len(set_a) == 0:
                    print "if you see this, something probably goes wrong"
                    return None
                res_li.append(len(set_a & set_s) / float(len(set_a)))
            return res_li

        def exceed_weights(w, max_w):
            for n in xrange(len(w)):
                if w[n] > max_w[n]:
                    return True
            return False

        def best_value(p, q, res):
            if p == 0:
                return 0
            _, value, w_set = self.items[p - 1]
            weight = get_weight(w_set & res, self.s_sets)
            if exceed_weights(weight, q):
                return best_value(p - 1, q, res)
            else:
                return max(best_value(p - 1, q, res),
                           best_value(p - 1, q, res & w_set) + value)

        j = self.max_weights
        result = []
        res_set = set(self.u_set)
        items = self.items
        for i in xrange(len(items), 0, -1):
            if best_value(i, j, res_set) != best_value(i - 1, j, res_set):
                # result.append(items[i - 1])
                result.append(items[i - 1][0])
                res_set &= items[i - 1][2]
                # print best_value(i - 1, j, res), result, res
                # j -= items[i - 1][1]
        result.reverse()
        return best_value(len(items), self.max_weights, set(self.u_set)), result

    def greedy_solver(self):
        def get_weight(set_a, s_sets):
            result = list()
            for set_s in s_sets:
                if len(set_a) == 0:
                    print "if you see this, something probably goes wrong"
                    return None
                result.append(len(set_a & set_s) / float(len(set_a)))
            # print len(result), result,
            return result

        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def find_max(l, cs):
            ratio = lambda p, w, c: p / float(sum([j / float(c[i]) for i, j in enumerate(w)]))
            first = True
            max_pw = -1
            sel = -1
            g_weights = list()
            for i in l:
                weights = get_weight(self.items[i][2] & cs, self.s_sets)
                pw = ratio(self.items[i][1], weights, self.max_weights)
                if pw < 0:
                    print "miracle"
                if first:
                    first = False
                    max_pw = pw
                    sel = i
                    g_weights = weights
                    continue
                if max_pw < 0:
                    if pw < max_pw:
                        sel = i
                        max_pw = pw
                        g_weights = weights
                elif pw > max_pw:
                    sel = i
                    max_pw = pw
                    g_weights = weights
            return sel, g_weights

        li = range(len(self.items))
        c_set = set(self.u_set)
        res = []
        best_value = 0
        while li:
            choose, new_weight = find_max(li, c_set)
            if not exceed_weights(new_weight, self.max_weights):
                res.append(self.items[choose][0])
                c_set &= self.items[choose][2]
                best_value += self.items[choose][1]
            li.pop(li.index(choose))
        return best_value, res


def test():
    # items = [(0, 4, 12), (1, 2, 1), (2, 6, 4), (3, 1, 1), (4, 2, 2)]
    items = [(0, 40, 2), (1, 50, 3.14), (2, 100, 1.98), (3, 95, 5), (4, 30, 3)]
    kp = Knapsack(items, 10)
    print kp.sorted_items
    print kp.dp_solver()
    print kp.bnb_solver()

    m_items = [(0, 5, (1, 2, 3)), (1, 3, (3, 1, 1)), (2, 8, (5, 1, 1)), (3, 2, (2, 2, 2)),
               (4, 4, (1, 5, 1)), (5, 10, (6, 2, 3))]
    m_weights = [9, 10, 7]
    mkp = MultiDimensionalKnapsack(m_items, m_weights)
    print mkp.sorted_items
    print mkp.dp_solver()
    print mkp.bnb_solver()
    print mkp.greedy_solver('scale')
    print mkp.greedy_solver('direct')

if __name__ == '__main__':
    test()
