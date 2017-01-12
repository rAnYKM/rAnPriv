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

import logging
import networkx as nx
import numpy as np
import pandas

from ran_kp import MultiDimensionalKnapsack, SetKnapsack, NetKnapsack

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class RPGraph:
    def __build_net(self, node, edge):
        """A snippet func for generating a NetworkX Graph
        :param node: node list
        :param edge: edge list
        :return: nx.Graph/nx.DiGraph
        """
        if self.is_directed:
            net = nx.DiGraph()
        else:
            net = nx.Graph()
        net.add_nodes_from(node)
        net.add_edges_from(edge)
        return net

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
        for j in attack_res.itervalues():
            all_number += j
        # if len(ctr) > 0:
        #     logging.debug("(exposed nodes) exceed number: %d" % (len(ctr)))
        #     print ctr
        return attack_res, np.average(all_number), sum(ctr) / float(ttn)

    def get_spd(self, node, secrets):
        """calculate the self privacy disclosure (spd) rate of the given node
        :param node: string
        :param secrets: list
        :return: dict
        """
        node_attr = [attr for attr in self.attr_net.neighbors_iter(node) if attr not in secrets]
        spd = {}
        for secret in secrets:
            s_set = set(self.attr_net.neighbors(secret))
            a_set = set(self.soc_net.nodes())
            for attr in node_attr:
                a_set &= set(self.attr_net.neighbors(attr))
            psa =  len(s_set & a_set)
            pa = len(a_set)
            spd[node] = psa/float(pa)
        return spd

    def __init__(self, soc_node, attr_node, soc_edge, attr_edge, is_directed=False):
        self.is_directed = is_directed
        self.soc_node = soc_node
        self.attr_node = attr_node
        self.soc_edge = soc_edge
        self.attr_edge = attr_edge
        self.soc_net = self.__build_net(soc_node, soc_edge)
        self.attr_net = self.__build_net(soc_node + attr_node, attr_edge)
        self.soc_attr_net = self.__build_net(soc_node + attr_node, soc_edge + attr_edge)
        logging.debug('[RPGraph] RAN built: %d actors, %d edges, %d attributes and %d links'
                      % (self.soc_net.number_of_nodes(), self.soc_net.number_of_edges(),
                         self.attr_net.number_of_nodes(), self.attr_net.number_of_edges()))
