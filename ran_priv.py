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
import logging
import numpy as np
from scipy import sparse
import networkx as nx
from networkx.algorithms import bipartite
import pandas
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
            feature = [node for node in self.soc_attr_net.neighbors(soc)
                       if node[0] == 'a' and node not in secrets[soc]]
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
    # ==============================

    # ==== Privacy Protection Algorithms ====
    def __get_max_weights(self, secrets, epsilon):
        """
        epsilon to theta
        :param secrets:
        :param epsilon:
        :return:
        """
        pass

    def v_knapsack_mask(self, secrets, price, epsilon, mode='greedy', p_mode='single'):
        """ knapsack-like solver
        :param secrets: {soc_node: [attr_node]}
        :param price: {attr_node: value} or {soc_node: {attr_node: value}}
        :param epsilon: float
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
                eps = epsilon[n]
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
                    val, sel = VecKnapsack(len(soc_node), s_arrays, items, eps).dp_solver()
                    tmp_res.append((val, sel))
                elif mode == 'greedy':
                    val, sel = VecKnapsack(len(soc_node), s_arrays, items, eps).greedy_solver()
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
        logging.debug('attr_array time: %f s' % (time.time() - t0))
        logging.debug('[RPGraph] RPGraph built: %d actors, %d edges, %d attributes and %d links'
                      % (self.soc_net.number_of_nodes(), self.soc_net.number_of_edges(),
                         self.attr_net.number_of_nodes() - self.soc_net.number_of_nodes(),
                         self.attr_net.number_of_edges()))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
