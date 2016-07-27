# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ran_graph.py
# Date: June. 18, 2016


import networkx as nx
import numpy as np
import logging
import operator
from random import Random
from ran_knapsack import knapsack
from ran_kp import MultiDimensionalKnapsack, SetKnapsack, NetKnapsack

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class RanGraph:
    """
    A Ran Social Network Graph Involves the following three sub-graphs:
    social network graph (social nodes, also actors)
    social attribute network graph (social nodes with attribute nodes)
    attribute network graph (attribute nodes)

    *ATTENTION: PLEASE AVOID DUPLICATE SOCIAL NODE AND ATTRIBUTE NAME*

    Social Node Naming:
    Simply use its original ID

    Attribute Naming:
    'a' + ABBR. of PATH + feature name
        Facebook example:
        education start_date 30 => aense_30
        Google+ example:
        job_title software => aje_software
    """

    @staticmethod
    def __conditional_prob(set_a, set_b):
        """
        return the conditional probability of two sets P(a|b)
        :param set_a: set
        :param set_b: set
        :return: float
        """
        if len(set_b) == 0:
            return 0
        return len(set_a & set_b) / float(len(set_b))

    @staticmethod
    def __joint_prob(set_a, set_b, set_u):
        """
        return the joint probability of two sets P(a,b) in the set_u where the elements in set_a
        and set_b are supposed to be in the set_u. Otherwise, return 0.
        :param set_a: set
        :param set_b: set
        :param set_u: set
        :return: float
        """
        if set_a <= set_u and set_b <= set_u:
            return len(set_a & set_b) / float(len(set_u))
        else:
            return 0

    @staticmethod
    def __conditional_entropy(set_a, set_b):
        """
        return the conditional entropy of two sets H(a|b). It means the uncertainty of a given b.
        :param set_a: set
        :param set_b: set
        :return: float
        """
        return np.log2(len(set_a & set_b) / float(len(set_b)))

    def __build_soc_net(self, soc_node, soc_edge):
        if self.is_directed:
            net = nx.DiGraph()
        else:
            net = nx.Graph()
        net.add_nodes_from(soc_node)
        net.add_edges_from(soc_edge)
        return net

    def __build_attr_net(self):
        attr_net = nx.Graph()
        nodes = [node for node in self.soc_attr_net.nodes() if node[0] == 'a']
        attr_net.add_nodes_from(nodes)
        for ns in nodes:
            for nd in nodes:
                if ns == nd:
                    continue
                elif not attr_net.has_edge(ns, nd):
                    # Calculate the correlation between two attribute nodes
                    # Jaccard Coefficient
                    neighbor_s = set(self.soc_attr_net.neighbors(ns))
                    neighbor_d = set(self.soc_attr_net.neighbors(nd))
                    if len(neighbor_s | neighbor_d) == 0:
                        cor = 0
                    else:
                        cor = len(neighbor_s & neighbor_d) / float(len(neighbor_s | neighbor_d))
                    if cor > 0.0:
                        attr_net.add_edge(ns, nd, {'weight': cor})
        return attr_net

    def __build_di_attr_net(self):
        attr_net = nx.DiGraph()
        nodes = [node for node in self.soc_attr_net.nodes() if node[0] == 'a']
        attr_net.add_nodes_from(nodes)
        for ns in nodes:
            for nd in nodes:
                if ns == nd:
                    continue
                elif not attr_net.has_edge(ns, nd):
                    # Calculate the correlation between two attribute nodes
                    # Conditional Probability
                    neighbor_s = set(self.soc_attr_net.neighbors(ns))
                    neighbor_d = set(self.soc_attr_net.neighbors(nd))
                    cor1 = self.__conditional_prob(neighbor_d, neighbor_s)
                    cor2 = self.__conditional_prob(neighbor_s, neighbor_d)
                    if cor1 > 0.0:
                        attr_net.add_edge(ns, nd, {'weight': cor1})
                    if cor2 > 0.0:
                        attr_net.add_edge(nd, ns, {'weight': cor2})
        return attr_net

    def __build_soc_attr_net(self, soc_node, attr_node, soc_edge, attr_edge):
        if self.is_directed:
            net = nx.DiGraph()
        else:
            net = nx.Graph()
        net.add_nodes_from(soc_node + attr_node)
        net.add_edges_from(soc_edge + attr_edge)
        return net

    def attribute_correlation(self, source, destination):
        """
        Calculate the correlation between source and destination attributes
        :param source: string
        :param destination: string
        :return: float
        """
        neighbor_s = set(self.soc_attr_net.neighbors(source))
        neighbor_d = set(self.soc_attr_net.neighbors(destination))
        return len(neighbor_s & neighbor_d) / float(len(neighbor_s | neighbor_d))

    def utility_measure(self, secrets, prices, mode='single'):
        scores = dict()
        for n in self.soc_net.nodes():
            fn = [i for i in self.soc_attr_net.neighbors(n)
                  if i[0] == 'a' and i not in secrets[n]]
            if mode == 'single':
                score = sum([prices[i] for i in fn])
            else:
                score = sum([prices[n][i] for i in fn])
            scores[n] = score
        return scores, sum(scores.itervalues())

    def relation_utility_measure(self, prices):
        scores = list()
        for u, v in self.soc_net.edges():
            if (u, v) in prices:
                scores.append(prices[(u, v)])
            elif (v, u) in prices:
                scores.append(prices[(v, u)])
            else:
                print "something may go wrong here", u, v
        return scores, sum(scores)

    def obtain_set(self, features):
        set_r = set(self.soc_net.nodes())
        for f in features:
            set_r &= set([n for n in self.soc_attr_net.neighbors(f)])
        return set_r

    def random_sampling(self, sampling=0.1):
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        a = Random()
        soc_edge = [edge for edge in self.soc_edge if a.random() <= sampling]
        attr_edge = [edge for edge in self.attr_edge
                     if a.random() <= sampling]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Random Sampling: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        logging.debug("Random Sampling: %d/%d social relations removed" %
                      (len(self.soc_edge) - len(soc_edge), len(self.soc_edge)))
        return new_ran

    def random_masking(self, secrets, mask_ratio):
        """
        return a sub graph with random mask algorithm for several secrets
        :param secrets: dict
        :param mask_ratio: float
        :return: RanGraph
        """
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        a = Random()
        soc_edge = [edge for edge in self.soc_edge if a.random() >= mask_ratio]
        attr_edge = [edge for edge in self.attr_edge
                     if edge[1] in secrets[edge[0]] or (len(secrets[edge[0]]) != 0 and a.random() >= mask_ratio)]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("Random Masking: %d/%d attribute edges removed" %
                      (len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        logging.debug("Random Masking: %d/%d social relations removed" %
                      (len(self.soc_edge) - len(soc_edge), len(self.soc_edge)))
        return new_ran

    def adv_random_masking(self, secrets, mask_ratio, mode='off'):
        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def exceed_all_weights(cur_graph):
            for nod, sec in secrets.iteritems():
                for index, je in enumerate(sec):
                    nei = cur_graph.neighbors(nod)
                    weight = self.prob_given_neighbor(je, nei)
                    if weight > mask_ratio[nod][index]:
                        return True
            return False


        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        soc_edge = []
        attr_edge = []
        deleted = []
        for n in soc_node:
            secret = secrets[n]
            if len(secret) == 0:
                attr_edge += [(n, attr) for attr in self.soc_attr_net.neighbors(n) if attr[0] == 'a']
                soc_edge += [(n, soc) for soc in self.soc_net.neighbors(n)]
            else:
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a' and i not in secrets[n]]
                weights = [self.prob_given_feature(s, fn) for s in secret]
                a = Random()
                while exceed_weights(weights, mask_ratio[n]) and fn:
                    fn.pop(int(a.random() * len(fn)))
                    weights = [self.prob_given_feature(s, fn) for s in secret]
                # Social Relations
                if mode == 'on':
                    sl = [i for i in self.soc_net.neighbors(n) if i not in deleted]
                    weights = [self.prob_given_neighbor(s, sl) for s in secret]
                    while exceed_weights(weights, mask_ratio[n]) and sl:
                        sl.pop(int(a.random() * len(sl)))
                        weights = [self.prob_given_neighbor(s, sl) for s in secret]
                    deleted += [(n, soc) for soc in self.soc_net.neighbors(n) if soc not in sl]
                    deleted += [(soc, n) for soc in self.soc_net.neighbors(n) if soc not in sl]
                    soc_edge += [(n, soc) for soc in sl if soc not in deleted]
                else:
                    deleted = []
                attr_edge += [(n, attr) for attr in fn]
                attr_edge += [(n, s) for s in secret]
        soc_edge = [edge for edge in soc_edge if edge not in deleted]
        # a = Random()
        # if mode == 'on':
        #     proc_edge = []
        #     for edge in self.soc_net.edges():
        #         u = edge[0]
        #         v = edge[1]
        #         if len(secrets[u]) == 0 and len(secrets[v]) == 0:
        #             soc_edge.append(edge)
        #         else:
        #             proc_edge.append(edge)
        #     cur_graph = nx.Graph()
        #     cur_graph.add_edges_from(proc_edge)
        #     print len(proc_edge)
        #     while(cur_graph.number_of_edges() != 0 and exceed_all_weights(cur_graph)):
        #         ed = proc_edge.pop(int(a.random()*len(proc_edge)))
        #         cur_graph.remove_edge(ed[0], ed[1])
        #     soc_edge += proc_edge
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Random Masking: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        logging.debug("Random Masking: %d/%d social relations removed" %
                      (len(self.soc_edge) - new_ran.soc_net.number_of_edges(), len(self.soc_edge)))
        return new_ran, attr_conceal / float(len(self.attr_edge)), (len(self.soc_edge) - new_ran.soc_net.number_of_edges()) / float(
            len(self.soc_edge))

    def nb_masking(self, secrets, mask_ratio, mode='off'):
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        soc_edge = []
        attr_edge = []

        def exceed_weights(w, max_w):
            for i in xrange(len(w)):
                if w[i] > max_w[i]:
                    return True
            return False

        def exceed_all_weights(cur_graph):
            for nod, sec in secrets.iteritems():
                for index, je in enumerate(sec):
                    nei = cur_graph.neighbors(nod)
                    weight = self.prob_given_neighbor(je, nei)
                    if weight > mask_ratio[nod][index]:
                        return True
            return False

        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        soc_edge = []
        attr_edge = []
        deleted = []
        for n in soc_node:
            secret = secrets[n]
            if len(secret) == 0:
                attr_edge += [(n, attr) for attr in self.soc_attr_net.neighbors(n) if attr[0] == 'a']
                soc_edge += [(n, soc) for soc in self.soc_net.neighbors(n)]
            else:
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a' and i not in secrets[n]]
                weights = [self.prob_given_feature(s, fn) for s in secret]
                sw = [sum([self.prob_given_feature(ff, [s])*0.1 + self.prob_given_feature(s, [ff])*0.9 for s in secret])
                      for ff in fn]
                while exceed_weights(weights, mask_ratio[n]) and fn:
                    index, value = max(enumerate(sw), key=operator.itemgetter(1))
                    fn.pop(index)
                    sw.pop(index)
                    weights = [self.prob_given_feature(s, fn) for s in secret]
                attr_edge += [(n, attr) for attr in fn]
                attr_edge += [(n, s) for s in secret]
                if mode == 'on':
                    sl = [i for i in self.soc_net.neighbors(n) if i not in deleted]
                    weights = [self.prob_given_neighbor(s, sl) for s in secret]
                    sw = [sum([self.prob_given_neighbor(s, [soc]) for s in secret])
                          for soc in sl]
                    while exceed_weights(weights, mask_ratio[n]) and sl:
                        index, value = max(enumerate(sw), key=operator.itemgetter(1))
                        sl.pop(index)
                        sw.pop(index)
                        weights = [self.prob_given_neighbor(s, sl) for s in secret]
                    deleted += [(n, soc) for soc in self.soc_net.neighbors(n) if soc not in sl]
                    deleted += [(soc, n) for soc in self.soc_net.neighbors(n) if soc not in sl]
                    soc_edge += [(n, soc) for soc in sl if soc not in deleted]
                else:
                    deleted = []
                attr_edge += [(n, attr) for attr in fn]
                attr_edge += [(n, s) for s in secret]
        soc_edge = [edge for edge in soc_edge if edge not in deleted]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Random Masking: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        logging.debug("Random Masking: %d/%d social relations removed" %
                      (len(self.soc_edge) - new_ran.soc_net.number_of_edges(), len(self.soc_edge)))
        return new_ran, attr_conceal / float(len(self.attr_edge)), (
        len(self.soc_edge) - new_ran.soc_net.number_of_edges()) / float(
            len(self.soc_edge))

    def random_mask(self, secret, mask_ratio=0.1):
        """
        return a sub graph with random mask algorithm (for single secret)
        :param secret: string
        :param mask_ratio: float
        :return: RanGraph
        """
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        a = Random()
        soc_edge = [edge for edge in self.soc_edge if a.random() >= mask_ratio]
        attr_edge = [edge for edge in self.attr_edge
                     if edge[1] == secret or (self.soc_attr_net.has_edge(edge[0], secret) and a.random() >= mask_ratio)]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Random Masking: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        logging.debug("Random Masking: %d/%d social relations removed" %
                      (len(self.soc_edge) - len(soc_edge), len(self.soc_edge)))
        return new_ran

    def knapsack_mask(self, secret, epsilon=0.5):
        """
        return a sub graph with knapsack mask algorithm
        :param secret: string
        :param epsilon: float
        :return: RanGraph
        """
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        soc_edge = self.soc_edge
        attr_edge = []
        w_set = set([n for n in self.soc_attr_net.neighbors(secret)])
        for n in soc_node:
            if not self.soc_attr_net.has_edge(n, secret):
                attr_edge += [(n, attr) for attr in self.soc_attr_net.neighbors(n) if attr[0] == 'a']
            else:
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a']
                feat = [(1, set(self.soc_attr_net.neighbors(i))) for i in self.soc_attr_net.neighbors(n)
                        if i[0] == 'a']
                val, sel = knapsack(feat, epsilon, w_set, set(self.soc_net.nodes()))
                attr_edge += [(n, fn[i]) for i in sel]
                attr_edge.append((n, secret))
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("Knapsack Masking: %d/%d attribute edges removed"
                      % (len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        return new_ran

    def mutual_information(self, attr_a, attr_b):
        """
        Return the mutual information between two attribtues
        :param attr_a: string
        :param attr_b: string
        :return: float
        """
        w_set = set(self.soc_net.nodes())
        a_set = set([n for n in self.soc_attr_net.neighbors(attr_a) if n[0] != 'a'])
        b_set = set([n for n in self.soc_attr_net.neighbors(attr_b) if n[0] != 'a'])
        value = self.__conditional_prob(a_set, b_set)
        value /= len(a_set) / float(len(w_set))
        return np.log2(value)

    def normalized_mutual_information(self, attr_a, attr_b, mode='ran'):
        """
        Return the mutual information between two attribtues
        :param attr_a: string
        :param attr_b: string
        :param mode: string
        :return: float
        """
        w_set = set(self.soc_net.nodes())
        a_set = set([n for n in self.soc_attr_net.neighbors(attr_a) if n[0] != 'a'])
        b_set = set([n for n in self.soc_attr_net.neighbors(attr_b) if n[0] != 'a'])
        if w_set & a_set != a_set:
            logging.error("set a is not the subset of the whole set")
        if mode == 'ran':
            v = -np.log2(len(a_set) / float(len(w_set)))
        else:
            v = np.sqrt(np.log2(len(a_set) / float(len(w_set))) * np.log2(len(b_set) / float(len(w_set))))
        return self.mutual_information(attr_a, attr_b) / v

    def d_knapsack_mask(self, secrets, price, epsilon, mode='greedy', p_mode='single'):
        """
        return a sub graph with satisfying epsilon-privacy
        :param secrets: dict
        :param epsilon: dict
        :return: RanGraph
        """
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                attr_edge += [(n, attr) for attr in self.soc_attr_net.neighbors(n) if attr[0] == 'a']
            else:
                eps = epsilon[n]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a' and i not in secrets[n]]
                items = list()
                for a in fn:
                    weight = tuple([self.mutual_information(a, s) for s in secrets[n]])
                    if p_mode == 'single':
                        items.append((a, price[a], weight))
                    else:
                        items.append((a, price[n][a], weight))
                    # 1 is the value
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                if mode == 'dp':
                    val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                else:
                    val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                attr_edge += [(n, attr[0]) for attr in sel]
                attr_edge += [(n, attr) for attr in secrets[n]]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking (%s): %d/%d attribute edges removed"
                      % (mode, len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        return new_ran, (len(self.attr_edge) - len(attr_edge)) / float(len(self.attr_edge))

    def s_knapsack_mask(self, secrets, price, epsilon, mode='dp', p_mode='single'):
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        tmp_res = []
        # oth_res = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                attr_edge += [(n, attr) for attr in self.soc_attr_net.neighbors(n) if attr[0] == 'a']
                # oth_res.append(sum([price[attr]
                #                     for attr in self.soc_attr_net.neighbors(n)
                #                     if attr[0] == 'a']))
            else:
                eps = epsilon[n]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a' and i not in secrets[n]]
                items = list()
                for a in fn:
                    weight = set([i for i in self.soc_attr_net.neighbors(a)])
                    if p_mode == 'single':
                        items.append((a, price[a], weight))
                    else:
                        items.append((a, price[n][a], weight))
                s_set = [set([i for i in self.soc_attr_net.neighbors(s)]) for s in secrets[n]]
                # 1 is the value
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                # val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                if mode == 'dp':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dp_solver()
                    tmp_res.append((val, sel))
                elif mode == 'greedy':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).greedy_solver()
                    tmp_res.append((val, sel))
                    # print val, sel
                elif mode == 'dual_greedy':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dual_greedy_solver()
                    val2, sel2 = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).greedy_solver()
                    print val, val2
                    print sel, sel2
                    tmp_res.append((val, sel))
                    # break
                else:
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dual_dp_solver()
                    val2, sel2 = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dp_solver()
                    print val, val2
                    print sel, sel2
                    tmp_res.append((val, sel))
                attr_edge += [(n, attr) for attr in sel]
                attr_edge += [(n, attr) for attr in secrets[n]]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        # sco = sum([i[0] for i in tmp_res]) + sum([i for i in oth_res])
        sco2 = new_ran.utility_measure(secrets, price, p_mode)
        logging.debug("s-Knapsack Masking (%s): %d/%d attribute edges removed"
                      % (mode, len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        logging.debug("score compare: %f" % (sco2[1]))
        return new_ran, (len(self.attr_edge) - len(attr_edge)) / float(len(self.attr_edge))

    def d_knapsack_relation(self, secrets, price, epsilon):
        """
        return a sub graph with satisfying epsilon-privacy for relation masking
        :param secrets: dict
        :param epsilon: dict
        :return: RanGraph
        """
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = []
        attr_edge = self.attr_edge
        deleted = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                soc_edge += [(n, node) for node in self.soc_net.neighbors(n) if (n ,node) not in deleted]
            else:
                eps = epsilon[n]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_net.neighbors(n) if (n, i) not in deleted]
                items = list()
                for a in fn:
                    weight = tuple([self.mutual_information(a, s) for s in secrets[n]])
                    items.append((a, price[a], weight))
                    # 1 is the value
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                deleted += [(n, soc[0]) for soc in items if soc not in sel]
                deleted += [(soc[0], n) for soc in items if soc not in sel]
                soc_edge += [(n, soc[0]) for soc in sel]
        soc_edge = [edge for edge in soc_edge if edge not in deleted]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking: %d/%d social relations removed"
                      % (len(self.soc_edge) - new_ran.soc_net.number_of_edges(), len(self.soc_edge)))
        return new_ran, (len(self.soc_edge) - len(soc_edge)) / float(len(self.soc_edge))

    def s_knapsack_relation(self, secrets, price, epsilon, mode='dp'):
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = []
        attr_edge = self.attr_edge
        tmp_res = []
        deleted = []
        # oth_res = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                soc_edge += [(n, soc) for soc in self.soc_net.neighbors(n) if (n, soc) not in deleted]
                # oth_res.append(sum([price[attr]
                #                     for attr in self.soc_attr_net.neighbors(n)
                #                     if attr[0] == 'a']))
            else:
                eps = epsilon[n]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_net.neighbors(n) if (n, i) not in deleted]
                items = list()
                for a in fn:
                    weight = set([i for i in self.soc_net.neighbors(a)])
                    items.append((a, price[a], weight))
                s_set = [set([i for i in self.soc_attr_net.neighbors(s)]) for s in secrets[n]]
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                # val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                if mode == 'dp':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dp_solver()
                    tmp_res.append((val, sel))
                elif mode == 'greedy':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).greedy_solver()
                    tmp_res.append((val, sel))
                    # print val, sel
                elif mode == 'dual_greedy':
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dual_greedy_solver()
                    # val2, sel2 = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).greedy_solver()
                    # print val, val2
                    # print sel, sel2
                    tmp_res.append((val, sel))
                    # break
                else:
                    val, sel = SetKnapsack(set(self.soc_net.nodes()), s_set, items, eps).dual_dp_solver()
                    tmp_res.append((val, sel))
                deleted += [(n, soc[0]) for soc in items if soc not in sel]
                deleted += [(soc[0], n) for soc in items if soc not in sel]
                soc_edge += [(n, attr) for attr in sel]
        # soc_edge = [edge for edge in soc_edge if edge not in deleted]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        # sco = sum([i[0] for i in tmp_res]) + sum([i for i in oth_res])
        sco2 = new_ran.relation_utility_measure(price)
        logging.debug("s-Knapsack Masking (%s): %d/%d social relations removed"
                      % (mode, len(self.soc_edge) - len(soc_edge), len(self.soc_edge)))
        logging.debug("score compare: %f" % (sco2[1]))
        return new_ran, (len(self.soc_edge) - len(soc_edge)) / float(len(self.soc_edge))

    def d_knapsack_relation_global(self, secrets, price, epsilon):
        """
        return a sub graph with satisfying epsilon-privacy for relation masking
        :param secrets: dict
        :param epsilon: dict
        :return: RanGraph
        """
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = []
        attr_edge = self.attr_edge
        # Serialize secrets and epsilon

        node_index = dict()
        new_eps = list()
        current = 0
        for node, secret in secrets.iteritems():
            eps = epsilon[node]
            if secret:
                node_index[node] = current
            else:
                node_index[node] = -1 # NO SECRET NODE
            for index, sec in enumerate(secret):
                new_eps.append(eps[index])
                current += 1

        items = list()
        for edge in self.soc_net.edges():
            u = edge[0]
            v = edge[1]
            if len(secrets[u]) == 0 and len(secrets[v]) == 0:
                soc_edge.append(edge)
                continue
            else:
                # Calculate weight
                weight = [0]*len(new_eps)
                for index, sec in enumerate(secrets[u]):
                    weight[node_index[u] + index] = self.mutual_information(v, sec)
                for index, sec in enumerate(secrets[v]):
                    weight[node_index[v] + index] = self.mutual_information(u, sec)
                item = (edge, price[edge], weight)
                items.append(item)
        # print items
        val, sel = MultiDimensionalKnapsack(items, new_eps).greedy_solver('scale')
        soc_edge += [choose[0] for choose in sel]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking: %d/%d social relations removed"
                      % (len(self.soc_edge) - new_ran.soc_net.number_of_edges(), len(self.soc_edge)))
        return new_ran, (len(self.soc_edge) - len(soc_edge)) / float(len(self.soc_edge))

    def s_knapsack_relation_global(self, secrets, price, epsilon):
        soc_node = self.soc_node
        attr_node = self.attr_node
        attr_edge = self.attr_edge
        soc_edge = list()
        # we want to globally consider a whole optimization problem
        # Get all max_weight and constraints
        # constraints is mapped by node
        items = list()
        for edge in self.soc_net.edges():
            if not (secrets[edge[0]] or secrets[edge[1]]):
                soc_edge.append(edge)
                continue
            item = (edge, price[edge])
            items.append(item)
        val, sel = NetKnapsack(self.soc_net, self.soc_attr_net, items, secrets, epsilon).greedy_solver()
        soc_edge += sel
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("N-Knapsack Masking: %d/%d social relations removed"
                      % (len(self.soc_edge) - len(soc_edge), len(self.soc_edge)))
        logging.debug("score compare: %f" % (val))
        return new_ran, (len(self.soc_edge) - len(soc_edge)) / float(len(self.soc_edge))

    def knapsack_relation(self, secret, epsilon=0.5):
        # WARNING: BE CAREFUL WITH USING THIS FUNCTION
        soc_node = self.soc_net.nodes()
        attr_node = self.attr_net.nodes()
        tmp_graph = self.soc_net
        attr_edge = self.attr_edge
        w_set = set([n for n in self.soc_attr_net.neighbors(secret)])
        for n in soc_node:
            if not self.soc_attr_net.has_edge(n, secret):
                continue
            else:
                fn = [i for i in self.soc_net.neighbors(n)]
                logging.debug("FN SIZE: %d" % len(fn))
                feat = [(1, set(self.soc_net.neighbors(i))) for i in self.soc_net.neighbors(n)]
                val, sel = knapsack(feat, epsilon, w_set, set(self.soc_net.nodes()))
                for i in range(len(fn)):
                    if i not in sel and tmp_graph.has_edge(n, fn[i]):
                        tmp_graph.remove_edge(n, fn[i])
                        # attr_edge.append((n, secret))
        new_ran = RanGraph(soc_node, attr_node, tmp_graph.edges(), attr_edge)
        logging.debug("Knapsack Relation Masking: %d/%d relations removed"
                      % (len(self.soc_edge) - tmp_graph.edges(), len(self.soc_edge)))
        return new_ran

    def secret_analysis(self, secret):
        """
        return the correlations dict of a given secret (private attribute)
        :param secret: string
        :return: dict
        """
        secret_related = self.di_attr_net.successors(secret)
        return {i: self.attribute_correlation(i, secret) for i in secret_related}

    def secret_disclosure_rate(self, secret, self_error_rate=1):
        """
        compare the new ran graph with the original one to obtain the disclosure_rate
        :return: float
        """
        pgf = []
        for soc in self.soc_net.nodes_iter():
            feature = [node for node in self.soc_attr_net.neighbors_iter(soc)
                       if node[0] == 'a' and node != secret]
            rate = self.prob_given_feature(secret, feature)
            if rate > self_error_rate:
                print "+1 exceeds"
            if self.soc_attr_net.has_edge(soc, secret):
                pgf.append(rate)
        pgn = []
        for soc in self.soc_net.nodes_iter():
            neighbor = [node for node in self.soc_net.neighbors_iter(soc)]
            rate = self.prob_given_feature(secret, neighbor)
            if rate > self_error_rate:
                print "+1 exceeds"
            if self.soc_attr_net.has_edge(soc, secret):
                pgn.append(rate)
        return pgf, pgn

    def self_disclosure_tree(self, secrets):
        """
        return a dict for each node's self disclosure rate
        :param secrets: dict
        :return: dict
        """
        attr_disclosure = dict()
        edge_disclosure = dict()
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            feature = [node for node in self.soc_attr_net.neighbors_iter(soc)
                       if node[0] == 'a' and node not in secrets[soc]]
            rates = {secret: self.prob_given_feature(secret, feature)
                     for secret in secrets[soc]}
            attr_disclosure[soc] = rates
            relation = [node for node in self.soc_net.neighbors_iter(soc)]
            r_rates = {secret: self.prob_given_feature(secret, relation)
                       for secret in secrets[soc]}
            edge_disclosure[soc] = r_rates
        return attr_disclosure, edge_disclosure

    def inference_attack(self, secrets, attack_graph, epsilon):
        """
        This function simulates the inference attack on several secrets from an attack_graph
        :param secrets: dict
        :param attack_graph: RanGraph
        :return: dict, float
        """
        attack_res = dict()
        ctr = list()
        ttn = 0
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            feature = [node for node in self.soc_attr_net.neighbors(soc)
                       if node[0] == 'a' and node not in secrets[soc]]
            att_feature = [node for node in feature if attack_graph.soc_attr_net.has_edge(soc, node)]
            # rates = {secret: self.prob_given_feature(secret, feature)
            #          for secret in secrets[soc]}
            att_rates = [attack_graph.prob_given_feature(secret, att_feature)
                         for secret in secrets[soc]]
            ctr += [(j - epsilon[soc][i])/float(epsilon[soc][i]) for i, j in enumerate(att_rates)
                    if j > epsilon[soc][i]]
            attack_res[soc] = att_rates
            ttn += len(att_rates)
        all_number = list()
        for j in attack_res.itervalues():
            all_number += j
        # if len(ctr) > 0:
        #     logging.debug("(exposed nodes) exceed number: %d" % (len(ctr)))
        #     print ctr
        return attack_res, np.average(all_number), sum(ctr)/float(ttn)

    def inference_attack_relation(self, secrets, attack_graph, epsilon):
        """
        This function simulates the inference attack on several secrets from an attack_graph
        VIA social relation information
        :param secrets: dict
        :param attack_graph: RanGraph
        :return: dict, float
        """
        attack_res = dict()
        ctr = list()
        ttn = 0
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            relation = [node for node in self.soc_net.neighbors_iter(soc)]
            att_feature = [node for node in relation if attack_graph.soc_net.has_edge(soc, node)]
            # rates = {secret: self.prob_given_feature(secret, feature)
            #          for secret in secrets[soc]}
            att_rates = [attack_graph.prob_given_feature(secret, att_feature)
                         for secret in secrets[soc]]
            ctr += [(j - epsilon[soc][i]) / float(epsilon[soc][i]) for i, j in enumerate(att_rates)
                    if j > epsilon[soc][i]]
            attack_res[soc] = att_rates
            ttn += len(att_rates)
        all_number = list()
        for j in attack_res.itervalues():
            all_number += j
        return attack_res, np.average(all_number), sum(ctr)/float(ttn)

    def secret_attack(self, secret, attack_graph):
        """
        This function simulate the single secret attack from the attack graph
        :param secret: string
        :param attack_graph: RanGraph
        :return: float
        """
        pgf = list()
        for soc in self.soc_net.nodes_iter():
            feature = [node for node in self.soc_attr_net.neighbors_iter(soc)
                       if node[0] == 'a' and node != secret]
            att_feature = [node for node in feature if attack_graph.soc_attr_net.has_edge(soc, node)]
            # rate = self.prob_given_feature(secret, feature)
            att_rate = attack_graph.prob_given_feature(secret, att_feature)
            if self.soc_attr_net.has_edge(soc, secret):
                pgf.append(att_rate)
        print pgf
        return np.average(pgf)

    def prob_given_feature(self, secret, feature):
        """
        Given a feature list, return the probability of owning a secret.
        :param secret: string
        :param feature: list
        :return: float
        """
        set_f = set(self.soc_net.nodes())
        first = True
        for f in feature:
            if first:
                set_f = set(self.soc_attr_net.neighbors(f))
                first = False
            else:
                set_f &= set(self.soc_attr_net.neighbors(f))
                if len(set_f) == 0:
                    return 0
        set_s = set(self.soc_attr_net.neighbors(secret))
        return self.__conditional_prob(set_s, set_f)

    def prob_given_neighbor(self, secret, neighbor):
        """
        Given a feature list, return the probability of owning a secret.
        :param secret: string
        :param neighbor: list
        :return: float
        """
        set_f = set(self.soc_net.nodes())
        first = True
        for f in neighbor:
            if first:
                set_f = set(self.soc_net.neighbors(f))
                first = False
            else:
                set_f &= set(self.soc_net.neighbors(f))
                if len(set_f) == 0:
                    return 0
        set_s = set(self.soc_attr_net.neighbors(secret))
        return self.__conditional_prob(set_s, set_f)

    def value_of_attribute(self, mode='equal'):
        values = dict()
        if mode == 'equal':
            for attr in self.attr_net.nodes():
                values[attr] = 1
        elif mode == 'unique':
            for attr in self.attr_net.nodes():
                values[attr] = 1/float(len(self.soc_attr_net.neighbors(attr)))
        elif mode == 'common':
            for node in self.soc_net.nodes():
                attrs = [soc for soc in self.soc_attr_net.neighbors_iter(node) if soc[0] == 'a']
                values[node] = dict()
                set_n = set(self.soc_net.neighbors(node))
                for attr in attrs:
                    set_a = set(self.soc_attr_net.neighbors(attr))
                    values[node][attr] = (len(set_n & set_a) + 1)/float(len(set_n) + 1)
        return values

    def value_of_relation(self, mode='equal'):
        values = dict()
        if mode == 'equal':
            for soc in self.soc_net.nodes():
                values[soc] = 1
        elif mode == 'unique':
            for soc in self.attr_net.nodes():
                values[soc] = 1 / float(len(self.soc_net.neighbors(soc)))
        return values

    def value_of_edge(self, mode='equal'):
        values = dict()
        if mode == 'equal':
            for edge in self.soc_net.edges():
                values[edge] = 1
        elif mode == 'Jaccard':
            for edge in self.soc_net.edges():
                u = edge[0]
                v = edge[1]
                u_set = set(self.soc_attr_net.neighbors(u))
                v_set = set(self.soc_attr_net.neighbors(v))
                values[edge] = len(u_set & v_set)/float(len(u_set | v_set))
        elif mode == 'AA':
            for edge in self.soc_net.edges():
                u = edge[0]
                v = edge[1]
                u_set = set(self.soc_attr_net.neighbors(u))
                v_set = set(self.soc_attr_net.neighbors(v))
                values[edge] = sum([np.log2(len([node for node in self.soc_attr_net.neighbors(w)
                                                 if node[0] != 'a']))
                                    for w in u_set & v_set])
        return values

    def __init__(self, soc_node, attr_node, soc_edge, attr_edge, is_directed=False):
        if is_directed:
            self.is_directed = True
        else:
            self.is_directed = False
        self.soc_node = soc_node
        self.attr_node = attr_node
        self.soc_edge = soc_edge
        self.attr_edge = attr_edge
        self.soc_net = self.__build_soc_net(soc_node, soc_edge)
        self.soc_attr_net = self.__build_soc_attr_net(soc_node, attr_node, soc_edge, attr_edge)
        """
        Attribute Network is built from Social Attribute Network
        Edges in Attribute Network represent the correlation
        Directed and Undirected attribute networks are both provided
        """
        self.attr_net = self.__build_attr_net()
        self.di_attr_net = self.__build_di_attr_net()
