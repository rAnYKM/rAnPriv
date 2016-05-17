# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: snap_google.py
# Date: May. 9, 2016

import os
import logging
import networkx as nx
from ranfig import load_ranfig


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SnapEgoNet:
    @staticmethod
    def __feat_process(feat):
        """
        EXAMPLE: 4 institution:AMC Theatres
        name: AMC Theatres
        no_cate: [4, 'institution']
        :param feat: String
        :return: Tuple
        """
        pairs = feat.strip().split(':')
        name = pairs[1]
        no_cate = pairs[0].split(' ')
        return name, no_cate[1]

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

    def __build_network(self):
        ego_net = nx.DiGraph()
        ego_net.add_edges_from(self.edges)
        ego_edges = [(self.root, vertex) for vertex in self.node.keys()]
        # Note: Because we DO NOT HAVE the feature list of followers, we comment this part
        # ego_edges += [(vertex, self.root) for vertex in self.followers]
        ego_net.add_edges_from(ego_edges)
        logging.debug('%d Nodes, %d Edges in the ego network.'
                      % (ego_net.number_of_nodes(), ego_net.number_of_edges()))
        return ego_net

    def __check_whether_in_node_feat(self, id_list):
        not_in_it = [vertex for vertex in id_list if vertex not in self.node]
        logging.debug('%d follower(s) not in the node feat list' % len(not_in_it))
        return not_in_it

    def __feat_name_list(self):
        with open(os.path.join(self.dir['SNAP'], self.root + '.featnames'), 'rb') as fp:
            feat_name = [self.__feat_process(line) for line in fp.readlines()]
            logging.debug('%d Feat(s) have been loaded.' % len(feat_name))
            return feat_name

    def __node_feat_list(self):
        with open(os.path.join(self.dir['SNAP'], self.root + '.feat'), 'rb') as fp:
            nodes = [self.__node_process(feat) for feat in fp.readlines()]
            node_dict = dict(nodes)
            logging.debug('%d User Feature List(s) have been loaded.' % len(nodes))
            return node_dict

    def __ego_feat_list(self):
        with open(os.path.join(self.dir['SNAP'], self.root + '.egofeat'), 'rb') as fp:
            li = fp.readline().strip('\r\n').split(' ')
            index = [num for num, value in enumerate(li) if value == '1']
            logging.debug('%d Ego Feature(s) have been loaded.' % len(index))
            return index

    def __edge_list(self):
        with open(os.path.join(self.dir['SNAP'], self.root + '.edges'), 'rb') as fp:
            edges = []
            follows_set = set()
            for line in fp.readlines():
                pairs = line.strip().split(' ')
                edges.append(pairs)
                follows_set.add(pairs[0])
                follows_set.add(pairs[1])
            logging.debug('%d Edge(s) have been loaded.' % len(edges))
            logging.debug('%d Ego Follow(s) have been loaded.' % len(follows_set))
            return edges, list(follows_set)

    def __follower_list(self):
        with open(os.path.join(self.dir['SNAP'], self.root + '.followers'), 'rb') as fp:
            followers = [line.strip() for line in fp.readlines()]
            logging.debug('%d Ego Follower(s) have been loaded.' % len(followers))
            return followers

    def get_ego_features(self):
        ego_features = [self.featname[feat] for feat in self.egofeat]
        return ego_features

    def get_network(self, mode='follow', label_with_feature='job_title'):
        """
        return a directed network with specific mode and labels
        :param mode: String 'friend', 'follow'
        :param label_with_feature: String Feature Category 'gender', 'institution', 'job_title', 'university', 'place'
        :return: nx.DiGraph
        """
        if mode == 'follow':
            network = self.network
            labels = list()
            for node in network.nodes_iter():
                if node == self.root:
                    feature = self.egofeat
                else:
                    feature = self.node[node]
                lab = [self.featname[f][0] for f in feature if self.featname[f][1] == label_with_feature]
                if len(lab) == 0:
                    lab = ['unlabeled']
                labels.append((node, {label_with_feature: ' '.join(lab)}))
            network.add_nodes_from(labels)
            nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-follow.gexf'))
            logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-follow.gexf'))
        elif mode == 'friend':
            network = nx.DiGraph()
            vertices = self.followers + [self.root]
            edges = [edge for edge in self.network.edges_iter()
                     if edge[0] in vertices and edge[1] in vertices]
            network.add_edges_from(edges)
            labels = list()
            for node in network.nodes_iter():
                if node == self.root:
                    feature = self.egofeat
                else:
                    feature = self.node[node]
                lab = [self.featname[f][0] for f in feature if self.featname[f][1] == label_with_feature]
                if len(lab) == 0:
                    lab = ['unlabeled']
                labels.append((node, {label_with_feature: ' '.join(lab)}))
            network.add_nodes_from(labels)
            nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))
            logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))

    def get_binary_label_network(self, features, mode='follow'):
        """
        return a directed network with specific mode and binary labels
        :param mode: String 'friend', 'follow'
        :param features: List of Int feature sequence numbers
        :return: nx.DiGraph
        """
        if mode == 'follow':
            network = self.network
            labels = list()
            for node in network.nodes_iter():
                if node == self.root:
                    feature = self.egofeat
                else:
                    feature = self.node[node]
                flag = False
                for f in feature:
                    if f in features:
                        flag = True
                        break
                labels.append((node, {'Binary Label': flag}))
            network.add_nodes_from(labels)
            nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-follow.gexf'))
            logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-follow.gexf'))
        elif mode == 'friend':
            network = nx.DiGraph()
            vertices = self.followers + [self.root]
            edges = [edge for edge in self.network.edges_iter()
                     if edge[0] in vertices and edge[1] in vertices]
            network.add_edges_from(edges)
            labels = list()
            for node in network.nodes_iter():
                if node == self.root:
                    feature = self.egofeat
                else:
                    feature = self.node[node]
                flag = False
                for f in feature:
                    if f in features:
                        flag = True
                        break
                labels.append((node, {'Binary Label': flag}))
            network.add_nodes_from(labels)
            nx.write_gexf(network, os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))
            logging.debug('Network Generated in %s' % os.path.join(self.dir['OUT'], self.root + '-ego-friend.gexf'))

    def attribute_distribution(self, category):
        """
        return a dict recording the attribute distribution
        :param category: string 'gender', 'institution', 'job_title', 'last_name', 'place', 'university'
        :return: dict
        """
        cate_feat = {num: 0 for num, item in enumerate(self.featname) if item[1] == category}
        cate_feat[-1] = 0  # Reserve for nodes without features in this category
        for n, features in self.node.iteritems():
            flag = False
            for f in features:
                if f in cate_feat:
                    cate_feat[f] += 1
                    flag = True
            if not flag:
                cate_feat[-1] += 1
        print cate_feat
        return cate_feat

    def __init__(self, ego_id):
        self.dir = load_ranfig()
        self.root = ego_id
        self.featname = self.__feat_name_list()
        self.node = self.__node_feat_list()
        self.egofeat = self.__ego_feat_list()
        self.edges, self.follows = self.__edge_list()
        self.followers = self.__follower_list()
        self.network = self.__build_network()
        # not_in_it = self.__check_whether_in_node_feat(self.followers)
        # not_in_it = self.__check_whether_in_node_feat(self.follows)


def main():
    user = SnapEgoNet('104226133029319075907')
    print(user.get_ego_features())
    # user.get_network(mode='friend', label_with_feature='institution')
    # user.get_binary_label_network(features=range(38, 45, 1), mode='friend')
    feats = [
            6,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,30,31,32,33,37,40,41,42,45,48,49,53,54,58,62,63,64,65,66,67,68,
            72,73,74,75,76,77,81,83,84,85,86,89,90,94,95,99,101,102,
            142, 143, 144, 145, 146, 157, 163, 179, 180, 181, 182, 243, 408, 409, 410, 411, 412, 413, 414, 415]
    # user.get_binary_label_network(features=feats, mode='follow')
    cate_feat = user.attribute_distribution('job_title')

if __name__ == '__main__':
    main()
