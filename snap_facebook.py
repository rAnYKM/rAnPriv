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
import matplotlib.pyplot as plt
import networkx as nx
import ran_tree as rt
from collections import Counter
from ranfig import load_ranfig
from ran_graph import RanGraph
from ran_knapsack import knapsack



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

    def __build_network(self):
        ego_net = nx.Graph()
        ego_net.add_edges_from(self.edges)
        ego_edges = [(self.root, vertex) for vertex in self.node.keys()]
        ego_net.add_edges_from(ego_edges)
        logging.debug('%d Nodes, %d Edges in the ego network.'
                      % (ego_net.number_of_nodes(), ego_net.number_of_edges()))
        return ego_net

    def __build_ran(self):
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
        soc_node = labels
        attr_node = attr_labels
        soc_edge = network.edges()
        ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge)
        return ran

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
        self.ran = self.__build_ran()


def main():
    fb_net = FacebookEgoNet('0')
    # fb_net.get_network()
    fb_net.attribute_stat()
    # print fb_net.get_ego_features()
    # fb_net.write_gexf_network(fb_net.ran, 'ran')
    # attr = [ver for ver in fb_net.ran.nodes() if ver[0] == 'a']
    # cor = {a: fb_net.attribute_correlation(a, 'aes39')
    # for a in attr if fb_net.attribute_correlation(a, 'aes39') > 0.0}
    # print cor
    # fb_net.write_gexf_network(fb_net.attr_net, 'attr')
    # print fb_net.ran.secret_analysis('aes50')
    print fb_net.ran.secret_disclosure_rate('aes50')
    att_ran = fb_net.ran.random_sampling(0.6)
    def_ran = fb_net.ran.random_mask('aes50', 0.6)
    print def_ran.secret_attack('aes50', att_ran)
    print fb_net.ran.secret_attack('aes50', att_ran)
    # print fb_net.ran.soc_attr_net['5']
    """Knapsack Test
    feat = [(1, set(fb_net.ran.soc_attr_net.neighbors(n))) for n in fb_net.ran.soc_attr_net.neighbors('50')
            if n[0] == 'a']
    print [n for n in fb_net.ran.soc_attr_net.neighbors('50')
            if n[0] == 'a']
    w_set = set([n for n in fb_net.ran.soc_attr_net.neighbors('aes50')])
    print knapsack(feat, 0.5, w_set, set(fb_net.ran.soc_net.nodes()))
    x = fb_net.ran.obtain_set(['al127', 'aet53', 'ag78', 'al118'])
    print x, [fb_net.ran.soc_attr_net.has_edge(n, 'aes50') for n in x]
    """
    # good_def_ran = fb_net.ran.knapsack_mask('aes50', 0.7)
    # edge_def_ran = fb_net.ran.knapsack_relation('aes50', 0.7)
    # print good_def_ran.secret_attack('aes50', att_ran)
    secret = dict()
    epsilon = dict()
    for i in fb_net.ran.soc_net.nodes():
        if fb_net.ran.soc_attr_net.has_edge(i, 'aes50'):
            secret[i] = ['aes50']
            epsilon[i] = [1.5]
        else:
            secret[i] = []
            epsilon[i] = []
    greedy = fb_net.ran.d_knapsack_mask(secret, epsilon)
    greedy2 = greedy.d_knapsack_relation(secret, epsilon)
    _, res = def_ran.inference_attack_relation(secret, att_ran)
    _, res2 = fb_net.ran.inference_attack_relation(secret, att_ran)
    print res, res2
    print greedy2.inference_attack(secret, att_ran)
    print greedy2.inference_attack_relation(secret, att_ran)

if __name__ == '__main__':
    main()
