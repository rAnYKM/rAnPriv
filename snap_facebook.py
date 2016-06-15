# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: snap_facebook.py
# Date: May. 18, 2016


import os
import logging
import networkx as nx
import ran_tree as rt
import numpy as np
from collections import Counter
from ranfig import load_ranfig


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FacebookEgoNet:

    @staticmethod
    def __abbr_attr(attr):
        abbr_l = [a[0] for a in attr]
        return ''.join(abbr_l)

    @staticmethod
    def __feat_process(line):
        """
        Split the raw data into an attribute
        Example:
        '12 education;classes;id;anonymized feature 12
        -> 12, ['education', 'classes', 'id']
        :param line: String
        :return: feature number, feature root
        """
        fields = line.strip().split(';')
        feat_name = fields[-1].strip('').replace('anonymized feature ', '')
        fields[0] = fields[0].split(' ')[1]
        cate_list = fields[:-1]
        return feat_name, cate_list

    @staticmethod
    def __node_process(feat):
        """
        ID [Binary Feature Vector]
        :param feat: String
        :return: Dict
        """
        li = feat.strip('\r\n').split(' ')
        uid = li[0]
        fea = li[1:]
        index = [num for num, value in enumerate(fea) if value == '1']
        return uid, index

    @staticmethod
    def __parse_path_attribute(act, path):
        a = act
        for n in path.split(','):
            a = a[n]
        return a

    @staticmethod
    def __conditional_prob(set_a, set_b):
        """
        return the conditional probability of two sets P(a|b)
        :param set_a: set
        :param set_b: set
        :return: float
        """
        return len(set_a & set_b)/float(len(set_b))

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
            return len(set_a & set_b)/float(len(set_u))
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
        return np.log2(len(set_a & set_b)/float(len(set_b)))

    def __attr_network(self):
        """
        Attribute Network involves a set of attribute nodes and correlation edges
        :return: nx.Graph
        """
        attr_net = nx.Graph()
        nodes = [node for node in self.ran.nodes() if node[0] == 'a']
        attr_net.add_nodes_from(nodes)
        for ns in nodes:
            for nd in nodes:
                if ns == nd:
                    continue
                elif not attr_net.has_edge(ns, nd):
                    # Calculate the correlation between two attribute nodes
                    # Jaccard Coefficient
                    neighbor_s = set(self.ran.neighbors(ns))
                    neighbor_d = set(self.ran.neighbors(nd))
                    cor = len(neighbor_s & neighbor_d) / float(len(neighbor_s | neighbor_d))
                    if cor > 0.0:
                        attr_net.add_edge(ns, nd, {'weight': cor})
        return attr_net

    def __attr_di_network(self):
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

    def __build_network(self):
        ego_net = nx.Graph()
        ego_net.add_edges_from(self.edges)
        ego_edges = [(self.root, vertex) for vertex in self.node.keys()]
        ego_net.add_edges_from(ego_edges)
        logging.debug('%d Nodes, %d Edges in the ego network.'
                      % (ego_net.number_of_nodes(), ego_net.number_of_edges()))
        return ego_net

    def __better_feature_structure(self):
        """
        we use rAnTree to represent the feature structure
        :return: rAnTree
        """
        feature_tree = rt.rAnTree()
        for feature, structure in self.featname:
            if 'id' in structure:
                structure.remove('id')
            feature_tree.add_path(structure)
        # feature_dict.display()
        return feature_tree

    def __better_node_feature(self):
        actors = dict()
        for n, feat_no in self.node.iteritems():
            # Initialize the feature tree of a single user
            actors[n] = self.category.build_dict()
            for i in feat_no:
                feat = self.featname[i]
                name = feat[0]
                path = feat[1]
                if 'id' in path:
                    path.remove('id')
                if len(path) == 1:
                    actors[n][path[0]] = name
                else:
                    actors[n][path[0]][path[1]] = name
        # Do not forget the root node
        actors[self.root] = self.category.build_dict()
        for i in self.egofeat:
            feat = self.featname[i]
            name = feat[0]
            path = feat[1]
            if 'id' in path:
                path.remove('id')
            if len(path) == 1:
                actors[n][path[0]] = name
            else:
                actors[n][path[0]][path[1]] = name
        return actors

    def __feat_name_list(self):
        with open(os.path.join(self.dir['FBOOK'], self.root + '.featnames'), 'rb') as fp:
            feat_name = [self.__feat_process(line) for line in fp.readlines()]
            logging.debug('%d Feat(s) have been loaded.' % len(feat_name))
            return feat_name

    def __node_feat_list(self):
        with open(os.path.join(self.dir['FBOOK'], self.root + '.feat'), 'rb') as fp:
            nodes = [self.__node_process(feat) for feat in fp.readlines()]
            node_dict = dict(nodes)
            logging.debug('%d User Feature List(s) have been loaded.' % len(nodes))
            return node_dict

    def __ego_feat_list(self):
        with open(os.path.join(self.dir['FBOOK'], self.root + '.egofeat'), 'rb') as fp:
            li = fp.readline().strip('\r\n').split(' ')
            index = [num for num, value in enumerate(li) if value == '1']
            logging.debug('%d Ego Feature(s) have been loaded.' % len(index))
            return index

    def __edge_list(self):
        with open(os.path.join(self.dir['FBOOK'], self.root + '.edges'), 'rb') as fp:
            edges = []
            follows_set = set()
            for line in fp.readlines():
                pairs = line.strip().split(' ')
                edges.append(pairs)
                follows_set.add(pairs[0])
                follows_set.add(pairs[1])
            logging.debug('%d Edge(s) have been loaded.' % len(edges))
            logging.debug('%d Ego Friend(s) have been loaded.' % len(follows_set))
            return edges, list(follows_set)

    def attribute_stat(self):
        paths = self.category.get_paths()
        for p in paths:
            li = [self.__parse_path_attribute(dic, p) for act, dic in self.actor.iteritems()]
            ctr = Counter(li)
            print p, ctr

    def attribute_correlation(self, source, destination):
        """
        Calculate the correlation between source and destination attributes
        Example: 'a100', 'a200'
        :param source: string
        :param destination: string
        :return: float
        """
        neighbor_s = set(self.ran.neighbors(source))
        neighbor_d = set(self.ran.neighbors(destination))
        return len(neighbor_s & neighbor_d)/float(len(neighbor_s | neighbor_d))

    def better_network(self):
        network = self.network
        labels = list()
        paths = self.category.get_paths()
        for node in network.nodes_iter():
            lab = dict()
            for p in paths:
                lab[p] = str(self.__parse_path_attribute(self.actor[node], p))
            labels.append((node, lab))
        network.add_nodes_from(labels)
        nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))
        logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))

    def prob_given_feature(self, secret, feature):
        """
        Given a feature list, return the probability of owning a secret.
        :param secret: string
        :param feature: list
        :return: float
        """
        set_f = set()
        first = True
        for f in feature:
            if first:
                set_f = set(self.ran.neighbors(f))
                first = False
            else:
                set_f &= set(self.ran.neighbors(f))
                if len(set_f) == 0:
                    return 0
        set_s = set(self.ran.neighbors(secret))
        return self.__conditional_prob(set_f, set_s)

    def get_ego_features(self):
        ego_features = [self.featname[feat] for feat in self.egofeat]
        return ego_features

    def get_network(self, label_with_feature='work'):
        """
        return a undirected network with specific labels
        :param label_with_feature: String Feature Category
        :return: nx.Graph
        """
        network = self.network
        labels = list()
        for node in network.nodes_iter():
            if node == self.root:
                feature = self.egofeat
            else:
                feature = self.node[node]
            lab = [self.featname[f][0] for f in feature if self.featname[f][1][0] == label_with_feature]
            if len(lab) == 0:
                lab = ['unlabeled']
            labels.append((node, {label_with_feature: ' '.join(lab)}))
        network.add_nodes_from(labels)
        nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))
        logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))

    def get_ran(self):
        network = nx.Graph(self.network)
        labels = [(node, {'lab': 'actor'}) for node in network.nodes_iter()]
        attr_labels = [('a' + self.__abbr_attr(feat[1]) + feat[0], {'lab': '.'.join(feat[1])})
                       for feat in self.featname]
        # Build Relational Attributes
        attr_edge = list()
        for node in network.nodes_iter():
            if node == self.root:
                feature = self.egofeat
            else:
                feature = self.node[node]
            for f in feature:
                attr_edge.append((node, 'a' + self.__abbr_attr(self.featname[f][1]) + self.featname[f][0]))
        network.add_edges_from(attr_edge)
        network.add_nodes_from(labels + attr_labels)
        # nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-ran.gexf'))
        # logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ran.gexf'))
        return network

    def secret_analysis(self, secret):
        """
        return the correlations dict of a given secret (private attribute)
        :param secret: string
        :return: dict
        """
        secret_related = self.attr_di_net.successors(secret)
        return {i: self.attribute_correlation(i, secret) for i in secret_related}

    def complete_disclosure_rate(self):
        """
        compare the new ran graph with the original one to obtain the disclosure_rate
        :return: float
        """
        # TODO: finish the complete disclosure rate calculation
        return 0

    def write_gexf_network(self, net, name):
        nx.write_gexf(net, os.path.join(self.dir['OUT'], self.root + '-ego-' + name + '.gexf'))
        logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-' + name + '.gexf'))

    def __init__(self, ego_id):
        self.dir = load_ranfig()
        self.root = ego_id
        self.featname = self.__feat_name_list()
        self.node = self.__node_feat_list()
        self.egofeat = self.__ego_feat_list()
        self.edges, self.friends = self.__edge_list()
        self.network = self.__build_network()
        self.category = self.__better_feature_structure()
        self.actor = self.__better_node_feature()
        self.ran = self.get_ran()
        self.attr_net = self.__attr_network()
        self.attr_di_net = self.__attr_di_network()


def main():
    fb_net = FacebookEgoNet('0')
    # fb_net.get_network()
    fb_net.attribute_stat()
    print fb_net.get_ego_features()
    # fb_net.write_gexf_network(fb_net.ran, 'ran')
    # attr = [ver for ver in fb_net.ran.nodes() if ver[0] == 'a']
    # cor = {a: fb_net.attribute_correlation(a, 'aes39')
    # for a in attr if fb_net.attribute_correlation(a, 'aes39') > 0.0}
    # print cor
    # fb_net.write_gexf_network(fb_net.attr_net, 'attr')
    print fb_net.secret_analysis('aes50')

if __name__ == '__main__':
    main()
