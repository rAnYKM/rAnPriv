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
from random import Random
from ran_knapsack import knapsack
from ran_kp import MultiDimensionalKnapsack

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
        value /= len(a_set)/float(len(w_set))
        return np.log2(value)

    def d_knapsack_mask(self, secrets, epsilon):
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
                # TODO: FINISH THE MULTIDIMENSIONAL KNAPSACK PROBLEM
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_attr_net.neighbors(n)
                      if i[0] == 'a' and i not in secrets[n]]
                items = list()
                for a in fn:
                    weight = tuple([self.mutual_information(a, s) for s in secrets[n]])
                    items.append((a, 1, weight))
                    # 1 is the value
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                attr_edge += [(n, attr[0]) for attr in sel]
                attr_edge += [(n, attr) for attr in secrets[n]]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking: %d/%d attribute edges removed"
                      % (len(self.attr_edge) - len(attr_edge), len(self.attr_edge)))
        return new_ran

    def d_knapsack_relation(self, secrets, epsilon):
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
                soc_edge += [(n, node) for node in self.soc_net.neighbors(n)]
            else:
                eps = epsilon[n]
                # Calculate the weight between secrets and attributes
                fn = [i for i in self.soc_net.neighbors(n) if (n, i) not in deleted]
                items = list()
                for a in fn:
                    weight = tuple([self.mutual_information(a, s) for s in secrets[n]])
                    items.append((a, 1, weight))
                    # 1 is the value
                # **WARNING** BE CAREFUL WHEN USING DP_SOLVER
                # val, sel = MultiDimensionalKnapsack(items, eps).dp_solver()
                val, sel = MultiDimensionalKnapsack(items, eps).greedy_solver('scale')
                deleted += [(n, soc[0]) for soc in items if soc not in sel]
                deleted += [(soc[0], n) for soc in items if soc not in sel]
                soc_edge += [(n, soc[0]) for soc in sel]
        new_ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        logging.debug("d-Knapsack Masking: %d/%d social relations removed"
                      % (len(self.soc_edge) - new_ran.soc_net.number_of_edges(), len(self.soc_edge)))
        return new_ran

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

    def secret_disclosure_rate(self, secret):
        """
        compare the new ran graph with the original one to obtain the disclosure_rate
        :return: float
        """
        pgf = []
        for soc in self.soc_net.nodes_iter():
            feature = [node for node in self.soc_attr_net.neighbors_iter(soc)
                       if node[0] == 'a' and node != secret]
            rate = self.prob_given_feature(secret, feature)
            if self.soc_attr_net.has_edge(soc, secret):
                pgf.append(rate)
        pgn = []
        for soc in self.soc_net.nodes_iter():
            neighbor = [node for node in self.soc_net.neighbors_iter(soc)]
            rate = self.prob_given_feature(secret, neighbor)
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

    def inference_attack(self, secrets, attack_graph):
        """
        This function simulates the inference attack on several secrets from an attack_graph
        :param secrets: dict
        :param attack_graph: RanGraph
        :return: dict, float
        """
        attack_res = dict()
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            feature = [node for node in self.soc_attr_net.neighbors_iter(soc)
                       if node[0] == 'a' and node not in secrets[soc]]
            att_feature = [node for node in feature if attack_graph.soc_attr_net.has_edge(soc, node)]
            # rates = {secret: self.prob_given_feature(secret, feature)
            #          for secret in secrets[soc]}
            att_rates = {secret: attack_graph.prob_given_feature(secret, att_feature)
                         for secret in secrets[soc]}
            attack_res[soc] = att_rates
        all_number = list()
        for j in attack_res.itervalues():
            for k in j.itervalues():
                all_number.append(k)
        return attack_res, np.average(all_number)

    def inference_attack_relation(self, secrets, attack_graph):
        """
        This function simulates the inference attack on several secrets from an attack_graph
        VIA social relation information
        :param secrets: dict
        :param attack_graph: RanGraph
        :return: dict, float
        """
        attack_res = dict()
        for soc in self.soc_net.nodes_iter():
            if len(secrets[soc]) == 0:
                # No secrets
                continue
            relation = [node for node in self.soc_net.neighbors_iter(soc)]
            att_feature = [node for node in relation if attack_graph.soc_net.has_edge(soc, node)]
            # rates = {secret: self.prob_given_feature(secret, feature)
            #          for secret in secrets[soc]}
            att_rates = {secret: attack_graph.prob_given_feature(secret, att_feature)
                         for secret in secrets[soc]}
            attack_res[soc] = att_rates
        all_number = list()
        for j in attack_res.itervalues():
            for k in j.itervalues():
                all_number.append(k)
        return attack_res, np.average(all_number)

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
