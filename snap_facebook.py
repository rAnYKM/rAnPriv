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
from ranfig import load_ranfig


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FacebookEgoNet:
    @staticmethod
    def __feat_process(line):
        """
        Split the raw data into an attribute
        Example:
        12 education;classes;id;anonymized feature 12
        -> 12,['education', 'classes', 'id']
        :param line:
        :return:
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

    def __init__(self, ego_id):
        self.dir = load_ranfig()
        self.root = ego_id
