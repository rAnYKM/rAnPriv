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
# Date: May. 18, 2016


import networkx as nx
import numpy as np


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
        education start_date 30 => aes30
        Google+ example:
        job_title software => ajsoftware
    """

    @staticmethod
    def __conditional_prob(set_a, set_b):
        """
        return the conditional probability of two sets P(a|b)
        :param set_a: set
        :param set_b: set
        :return: float
        """
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
                    cor = len(neighbor_s & neighbor_d) / float(len(neighbor_s | neighbor_d))
                    if cor > 0.0:
                        attr_net.add_edge(ns, nd, {'weight': cor})
        return attr_net

    def __build_di_attr_net(self):
        attr_net = nx.DiGraph()
        nodes = [node for node in self.ran.nodes() if node[0] == 'a']
        attr_net.add_nodes_from(nodes)
        for ns in nodes:
            for nd in nodes:
                if ns == nd:
                    continue
                elif not attr_net.has_edge(ns, nd):
                    # Calculate the correlation between two attribute nodes
                    # Conditional Probability
                    neighbor_s = set(self.ran.neighbors(ns))
                    neighbor_d = set(self.ran.neighbors(nd))
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

    def __init__(self, soc_node, attr_node, soc_edge, attr_edge, is_directed=False):
        if is_directed:
            self.is_directed = True
        else:
            self.is_directed = False
        self.soc_net = self.__build_soc_net(soc_node, soc_edge)
        self.soc_attr_net = self.__build_soc_attr_net(soc_node, attr_node, soc_edge, attr_edge)
        """
        Attribute Network is built from Social Attribute Network
        Edges in Attribute Network represent the correlation
        Directed and Undirected attribute networks are both provided
        """
        self.attr_net = self.__build_attr_net()
        self.di_attr_net = self.__build_di_attr_net()
