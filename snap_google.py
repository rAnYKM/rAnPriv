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
from ranfig import load_ranfig
import logging


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
            id, node = self.__node_process(fp.readline())
            logging.debug('%d Ego Feature(s) have been loaded.' % len(node))
            return node

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

    def __init__(self, ego_id):
        self.dir = load_ranfig()
        self.root = ego_id
        self.featname = self.__feat_name_list()
        self.node = self.__node_feat_list()
        self.egofeat = self.__ego_feat_list()
        self.edges, self.follows = self.__edge_list()
        self.followers = self.__follower_list()


def main():
    user = SnapEgoNet('100129275726588145876')


if __name__ == '__main__':
    main()
