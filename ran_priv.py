# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ran_priv.py
# Date: Jan. 11, 2017

from __future__ import division, print_function, absolute_import

import time
import operator
import logging
import numpy as np
import networkx as nx
import pandas as pd
from ran_kp import MultiDimensionalKnapsack, SetKnapsack, NetKnapsack, VecKnapsack

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class RPGraph:
    """
    Test Code
    >>> soc_node = [1, 2, 3, 4, 5]
    >>> attr_node = ['a', 'b', 'c', 'd']
    >>> soc_edge = [(1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 5)]
    >>> attr_edge = [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'c'), (3, 'd'), (4, 'a'), (4, 'd'), (5, 'c')]
    >>> G = RPGraph(soc_node, attr_node, soc_edge, attr_edge)
    >>> G.get_spd(2, ['a'])
    {'a': 1.0}
    """
    @staticmethod
    def __build_net(node, edge, is_directed=False):
        """A snippet func for generating a NetworkX Graph
        :param node: node list
        :param edge: edge list
        :return: nx.Graph/nx.DiGraph
        """
        if is_directed:
            net = nx.DiGraph()
        else:
            net = nx.Graph()
        net.add_nodes_from(node)
        net.add_edges_from(edge)
        return net

    @staticmethod
    def __build_bigraph(node_0, node_1, edge, is_directed=False):
        """A snippet func for generating a NetworkX BiGraph
        :param node_0: node 0 list
        :param node_1: node 1 list
        :param edge: edge list
        :return: nx.Graph
        """
        if is_directed:
            net = nx.DiGraph()
        else:
            net = nx.Graph()
        net.add_nodes_from(node_0, bipartite=0)
        net.add_nodes_from(node_1, bipartite=1)
        net.add_edges_from(edge)
        return net

    def __get_attr_array(self):
        """attribute-user array. {attribute: np.array([0, 0, 1, 1, 0])}
        :return: attribute - user array dict
        """
        attr_dict = {}
        sup_soc = {n: i for i, n in enumerate(self.soc_node)}
        for attr in self.attr_node:
            # The following code has a very poor performance
            # attr_dict[attr] = np.array([int(self.attr_net.has_edge(soc, attr)) for soc in self.soc_node])
            tmp_array = np.zeros(len(self.soc_node))
            for soc in self.attr_net.neighbors_iter(attr):
                tmp_array[sup_soc[soc]] = 1
            attr_dict[attr] = tmp_array
        return attr_dict

    def mutual_information(self, attr_a, attr_b):
        """
        Return the mutual information between two attribtues
        :param attr_a: string
        :param attr_b: string
        :return: float
        """
        vec_a = self.attr_array[attr_a]
        vec_b = self.attr_array[attr_b]
        value = len(self.soc_node) * vec_a.dot(vec_b.transpose()) / (vec_a.sum() * vec_b.sum())
        return np.log2(value)

    # ==== Important functions ====
    def prob_secret_on_attributes(self, secret, attributes):
        """Conditional probability of having a secret on several attributes
        :param secret: attr_node
        :param attributes: [attr_node]
        :return: float
        """
        s_array = self.attr_array[secret]
        a_array = np.ones(len(self.soc_node))
        for attr in attributes:
            a_array *= self.attr_array[attr]
        psa = int(s_array.dot(a_array.transpose()))
        pa = a_array.sum()
        return psa / pa
    # =============================

    # ==== Theoretical Analysis ====
    def get_spd(self, node, secrets):
        """calculate the self privacy disclosure (spd) rate of the given node in matrix mode
        :param node: string
        :param secrets: list
        :return: dict
        """
        node_attr = [attr for attr in self.attr_net.neighbors_iter(node) if attr not in secrets]
        spd = {}
        for secret in secrets:
            spd[secret] = self.prob_secret_on_attributes(secret, node_attr)
        return spd

    def inference_attack(self, secrets, attack_graph, epsilon):
        """Simulate the inference attack on several secrets from an attack_graph
        :param secrets: dict
        :param attack_graph: RanGraph
        :param epsilon: string
        :return: dict, float
        """
        attack_res = dict()
        ctr = list()
        ttn = 0
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            feature = [node for node in self.attr_net.neighbors(soc)
                       if node not in secrets[soc]]
            att_feature = [node for node in feature if attack_graph.soc_attr_net.has_edge(soc, node)]
            # rates = {secret: self.prob_given_feature(secret, feature)
            #          for secret in secrets[soc]}
            att_rates = [attack_graph.prob_given_feature(secret, att_feature)
                         for secret in secrets[soc]]
            ctr += [j - epsilon[soc][i] for i, j in enumerate(att_rates)
                    if j > epsilon[soc][i]]
            attack_res[soc] = att_rates
            ttn += len(att_rates)
        all_number = list()
        for j in iter(attack_res.values()):
            all_number += j
        # if len(ctr) > 0:
        #     logging.debug("(exposed nodes) exceed number: %d" % (len(ctr)))
        #     print ctr
        return attack_res, np.average(all_number), sum(ctr) / float(ttn)

    def infer_identify(self, secrets, attack_graph):
        """Inference {user - attributes - secrets}, re-identify user attributes in the attack_graph
        :param secrets: {soc_node: [attr_node]}
        :param attack_graph: RPGraph
        :return: result dict {soc_node: {secret: spd}}
        """
        result = {}
        for node, secret in secrets.items():
            node_attr = [attr for attr in self.attr_net.neighbors_iter(node) if attr not in secret]
            # TODO: Directly use the node_attr instead of attack_attr
            attack_attr = [attr for attr in node_attr if attack_graph.attr_net.has_edge(node_attr)]
            spd = {s: attack_graph.prob_secret_on_attributes(s, attack_attr) for s in secret}
            result[node] = spd
        return result

    def affected_attribute_number(self, secrets):
        """ Get the original number of attributes owned by the users with secrets
        :param secrets:
        :return:
        """
        attr_num = 0
        for n in self.soc_net.nodes_iter():
            if len(secrets[n]) == 0:
                # No secrets
                continue
            else:
                attr_num += len([i for i in self.attr_net.neighbors(n) if i not in secrets[n]])
        return attr_num
    # ==============================

    # ==== Privacy Protection Algorithms ====
    def __get_max_weight(self, secret, epsilon, delta):
        """
        epsilon to theta
        :param secret: attr_node
        :param epsilon: float
        :param delta: float
        :return: float
        """
        # Prior probability of secret
        prior = len(self.attr_net.neighbors(secret)) / len(self.soc_node)
        return np.exp(epsilon) * prior + delta

    def __get_max_weight_dkp(self, secret, epsilon, delta):
        """
        epsilon to theta
        :param secret: attr_node
        :param epsilon: float
        :param delta: float
        :return: float
        """
        former_weight = self.__get_max_weight(secret, epsilon, delta)
        prior = len(self.attr_net.neighbors(secret)) / len(self.soc_node)
        return np.log2(former_weight / prior)

    def naive_bayes_mask(self, secrets, epsilon, delta, factor=0.1):
        def exceed_weights(w, max_w):
            for wi in range(len(w)):
                if w[wi] > max_w[wi]:
                    return True
            return False

        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        for n in soc_node:
            secret = secrets[n]
            if len(secret) == 0:
                attr_edge += [(n, attr) for attr in self.attr_net.neighbors(n)]
            else:
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.attr_net.neighbors(n)
                      if i not in secrets[n]]
                weights = [self.prob_secret_on_attributes(s, fn) for s in secret]
                sw = [sum([self.prob_secret_on_attributes(ff, [s]) * factor +
                           self.prob_secret_on_attributes(s, [ff]) * (1 - factor)
                           for s in secret])
                      for ff in fn]
                mask_ratio = [self.__get_max_weight(s, epsilon, delta) for s in secret]
                while exceed_weights(weights, mask_ratio) and fn:
                    index, value = max(enumerate(sw), key=operator.itemgetter(1))
                    fn.pop(index)
                    sw.pop(index)
                    weights = [self.prob_secret_on_attributes(s, fn) for s in secret]
                attr_edge += [(n, attr) for attr in fn]
                attr_edge += [(n, s) for s in secret]
        new_rpg = RPGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Naive Bayes Masking: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        return new_rpg

    def entropy_mask(self, secrets, epsilon, delta):
        """ knapsack-like solver
        :param secrets: {soc_node: [attr_node]}
        :param epsilon: float
        :param delta: float
        :return: RPGraph
        """
        # TODO: Implementation
        def exceed_weights(w, max_w):
            for wi in range(len(w)):
                if w[wi] > max_w[wi]:
                    return True
            return False

        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        for n in soc_node:
            secret = secrets[n]
            if len(secret) == 0:
                attr_edge += [(n, attr) for attr in self.attr_net.neighbors(n)]
            else:
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.attr_net.neighbors(n)
                      if i not in secrets[n]]
                weights = [self.prob_secret_on_attributes(s, fn) for s in secret]
                sw = [sum([self.mutual_information(s, ff) for s in secret]) for ff in fn]
                mask_ratio = [self.__get_max_weight(s, epsilon, delta) for s in secret]
                while exceed_weights(weights, mask_ratio) and fn:
                    index, value = max(enumerate(sw), key=operator.itemgetter(1))
                    fn.pop(index)
                    sw.pop(index)
                    weights = [self.prob_secret_on_attributes(s, fn) for s in secret]
                attr_edge += [(n, attr) for attr in fn]
                attr_edge += [(n, s) for s in secret]
        new_rpg = RPGraph(soc_node, attr_node, soc_edge, attr_edge)
        attr_conceal = len(self.attr_edge) - len(attr_edge)
        logging.debug("Entropy Masking: %d/%d attribute edges removed" % (attr_conceal, len(self.attr_edge)))
        return new_rpg

    def d_knapsack_mask(self, secrets, price, epsilon, delta, mode='greedy', p_mode='single'):
        """ knapsack-like solver
        :param secrets: {soc_node: [attr_node]}
        :param price: {attr_node: value} or {soc_node: {attr_node: value}}
        :param epsilon: float
        :param delta: float
        :param mode: string
        :param p_mode: string
        :return: RPGraph
        """
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                attr_edge += [(n, attr) for attr in self.attr_net.neighbors(n)]
            else:
                max_weights = [self.__get_max_weight_dkp(secret, epsilon, delta) for secret in secrets[n]]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.attr_net.neighbors(n) if i not in secrets[n]]
                items = list()
                for a in fn:
                    weight = tuple([self.mutual_information(a, s) for s in secrets[n]])
                    if p_mode == 'single':
                        items.append((a, price[a], weight))
                    else:
                        items.append((a, price[n][a], weight))
                if mode == 'dp':
                    val, sel = MultiDimensionalKnapsack(items, max_weights).dp_solver()
                else:
                    val, sel = MultiDimensionalKnapsack(items, max_weights).greedy_solver('scale')
                attr_edge += [(n, attr[0]) for attr in sel]
                attr_edge += [(n, attr) for attr in secrets[n]]
        new_ran = RPGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking (%s): %d/%d attribute edges removed"
                      % (mode, len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        return new_ran, (len(self.attr_edge) - len(attr_edge)) / float(len(self.attr_edge))

    def v_knapsack_mask(self, secrets, price, epsilon, delta, mode='greedy', p_mode='single'):
        """ knapsack-like solver
        :param secrets: {soc_node: [attr_node]}
        :param price: {attr_node: value} or {soc_node: {attr_node: value}}
        :param epsilon: float
        :param delta: float
        :param mode: string
        :param p_mode: string
        :return: RPGraph
        """
        soc_node = self.soc_node
        attr_node = self.attr_node
        soc_edge = self.soc_edge
        attr_edge = []
        tmp_res = []
        for n in self.soc_net.nodes():
            if len(secrets[n]) == 0:
                attr_edge += [(n, attr) for attr in self.attr_net.neighbors(n)]
            else:
                # TODO: epsilon need to be calculated again
                max_weights = [self.__get_max_weight(secret, epsilon, delta) for secret in secrets[n]]
                # Calculate the weight between secrets and attributes
                original_attr = [i for i in self.attr_net.neighbors(n) if i not in secrets[n]]
                items = list()
                for attr in original_attr:
                    weight = self.attr_array[attr]
                    if p_mode == 'single':
                        items.append((attr, price[attr], weight))
                    else:
                        items.append((attr, price[n][attr], weight))
                s_arrays = [self.attr_array[s] for s in secrets[n]]
                if mode == 'dp':
                    val, sel = VecKnapsack(len(soc_node), s_arrays, items, max_weights).dp_solver()
                    tmp_res.append((val, sel))
                elif mode == 'greedy':
                    val, sel = VecKnapsack(len(soc_node), s_arrays, items, max_weights).greedy_solver()
                    tmp_res.append((val, sel))
                else:
                    pass
                attr_edge += [(n, attr) for attr in sel]
                attr_edge += [(n, attr) for attr in secrets[n]]
        new_ran = RPGraph(soc_node, attr_node, soc_edge, attr_edge)
        # sco = sum([i[0] for i in tmp_res]) + sum([i for i in oth_res])
        # sco2 = new_ran.utility_measure(secrets, price, p_mode)
        logging.debug("v-Knapsack Masking (%s): %d/%d attribute edges removed"
                      % (mode, len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        # logging.debug("score compare: %f" % (sco2[1]))
        return new_ran
    # =======================================

    def __init__(self, soc_node, attr_node, soc_edge, attr_edge, is_directed=False):
        self.is_directed = is_directed
        self.soc_node = soc_node
        # self.soc_node_sup = {n: i for i, n in enumerate(soc_node)}
        self.attr_node = attr_node
        # self.attr_node_sup = {n: i for i, n in enumerate(attr_node)}
        self.soc_edge = soc_edge
        self.attr_edge = attr_edge
        self.soc_net = self.__build_net(soc_node, soc_edge, self.is_directed)
        # bigraph row: soc_node, col: attr_node
        self.attr_net = self.__build_bigraph(soc_node, attr_node, attr_edge)
        # self.soc_attr_net = self.__build_net(soc_node + attr_node, soc_edge + attr_edge, self.is_directed)
        t0 = time.time()
        self.attr_array = self.__get_attr_array()
        # logging.debug('attr_array time: %f s' % (time.time() - t0))
        logging.debug('[RPGraph] RPGraph built: %d actors, %d edges, %d attributes and %d links'
                      % (self.soc_net.number_of_nodes(), self.soc_net.number_of_edges(),
                         self.attr_net.number_of_nodes() - self.soc_net.number_of_nodes(),
                         self.attr_net.number_of_edges()))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
