# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _
#    _____/   |  ____  / __ \_____(_)   __
#   / ___/ /| | / __ \/ /_/ / ___/ / | / /
#  / /  / ___ |/ / / / ____/ /  / /| |/ /
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/
#
# Script Name: snap_gpcomplete.py
# Date: Mar. 16, 2017

import os
import re
import logging
import time
import numpy as np
import pandas as pd
import networkx as nx
from ranfig import load_ranfig
from ran_priv import RPGraph
from ran_lab import attr_statistics, AttributeExperiment, RelationExperiment

WORD_IGNORE = ['.com', 'Inc', 'Corporation', 'Employed', 'employed', 'Company', 'Corps', 'Microsystems',
               '.tv', 'Media', 'Ventures', 'Android', 'Research', 'Interactive', 'News']
WORD_NONSENSE = ['and', 'also', 'all', 'are', 'but', 'being', 'for', 'from', 'every', 'guy', 'just', 'most',
                 'one', 'open', 'our', 'plus', 'really', 'see']
CATE_HEADER = ['gender_', 'inst_', 'job_', 'place_', 'univ_']
REMOVABLE = 'ReMoVe'
ALL_EDGES_FILE = 'gplus_combined.txt'

class GoogleEgo:
    def read_raw_featnames(self):
        with open(os.path.join(self.dirs['GPLUS'], self.ego + '.featnames')) as fp:
            lines = fp.readlines()
            raw_featnames = []
            for line in lines:
                # Process Line
                # 0 gender:1
                split_one = line.strip().split(':')
                attr = split_one[1]
                cate = split_one[0].split(' ')[1]
                raw_featnames.append((attr, cate))
            return  raw_featnames

    def read_raw_feat(self):
        with open(os.path.join(self.dirs['GPLUS'], self.ego + '.feat')) as fp:
            lines = fp.readlines()
            raw_feats = {}
            for line in lines:
                items = line.strip().split(' ')
                uid = items[0]
                feats = items[1:]
                feat_indices = [index for index, value in enumerate(feats) if value == '1']
                raw_feats[uid] = feat_indices
            return raw_feats

    def read_raw_self_feat(self):
        with open(os.path.join(self.dirs['GPLUS'], self.ego + '.egofeat')) as fp:
            lines = fp.readlines()[0]
            line = lines[0]
            items = line.strip().split(' ')
            feats = items
            feat_indices = [index for index, value in enumerate(feats) if value == '1']
            return feat_indices

    def __process_feat_name(self, feat_list):
        new_feat_list = []
        for feat, cate in feat_list:
            if cate == 'gender':
                result = CATE_HEADER[0] + feat
            elif cate == 'institution':
                for word in WORD_IGNORE:
                    feat = re.sub(word, '', feat)
                level_1 = feat.lower()
                level_2 = re.sub('[\W_]+', '', level_1)
                if len(level_2) < 2:
                    result = REMOVABLE
                else:
                    result = CATE_HEADER[1] + level_2
            elif cate == 'job_title':
                level_1 = feat.lower()
                level_2 = re.sub('[\W_]+', '', level_1)
                if len(level_2) < 3:
                    result = REMOVABLE
                else:
                    result = CATE_HEADER[2] + level_2
            elif cate == 'place':
                feat = feat.split(',')[0]
                level_1 = feat.lower()
                level_2 = re.sub('[\W_]+', '', level_1)
                if len(level_2) < 2:
                    result = REMOVABLE
                else:
                    result = CATE_HEADER[3] + level_2
            elif cate == 'university':
                level_1 = feat.lower()
                level_2 = re.sub('[\W_]+', '', level_1)
                if len(level_2) < 3:
                    result = REMOVABLE
                else:
                    result = CATE_HEADER[4] + level_2
            else:
                result = REMOVABLE
            new_feat_list.append(result)
        return new_feat_list

    def get_profiles(self):
        raw_featnames = self.read_raw_featnames()
        new_featnames = self.__process_feat_name(raw_featnames)
        raw_feats = self.read_raw_feat()
        raw_feats[self.ego] = self.read_raw_self_feat()
        profiles = {}
        for uid, feats in raw_feats.items():
            tmp_profile = set()
            for index in feats:
                feat_name = new_featnames[index]
                if feat_name == REMOVABLE:
                    continue
                else:
                    tmp_profile.add(feat_name)
            profiles[uid] = list(tmp_profile)
        return profiles

    def get_attr_nodes_edges(self, profiles):
        attr_edges = []
        attr_nodes = set()
        for uid, profile in profiles.items():
            for attr in profile:
                attr_edges.append((uid, attr))
                attr_nodes.add(attr)
        return list(attr_nodes), attr_edges

    def __build_attr_net(self, attr_edges):
        graph = nx.Graph()
        graph.add_edges_from(attr_edges)
        return graph

    def prune_useless_part(self, threshold=5):
        profiles = self.get_profiles()
        soc_nodes = profiles.keys()
        attr_nodes, attr_edges = self.get_attr_nodes_edges(profiles)
        attr_net = self.__build_attr_net(attr_edges)
        # select attr nodes
        useless_attr_nodes = []
        for attr in attr_nodes:
            if attr_net.degree(attr) < threshold:
                useless_attr_nodes.append(attr)
        attr_net.remove_nodes_from(useless_attr_nodes)
        useless_soc_nodes = []
        for soc in soc_nodes:
            if attr_net.degree(soc) < threshold:
                useless_soc_nodes.append(soc)
        attr_net.remove_nodes_from(useless_soc_nodes)
        logging.debug('[GPEgo] #%s, %d/%d social nodes, %d/%d attributes' % (self.ego,
                                                                            len(soc_nodes) - len(useless_soc_nodes),
                                                                            len(soc_nodes),
                                                                            len(attr_nodes) - len(useless_attr_nodes),
                                                                            len(attr_nodes)
                                                                            ))
        return attr_net

    def __init__(self, uid):
        self.ego = uid
        self.dirs = load_ranfig()


class GooglePlusNetwork:
    def first_run(self):
        pass

    def get_total_soc_graph(self, soc_nodes, filename=ALL_EDGES_FILE):
        count = 0
        graph = nx.DiGraph()
        with open(os.path.join(self.dirs['GPLUS'], filename), 'rb') as fp:
            for line in fp:
                edge = line.strip().split(' ')
                u, v = edge
                graph.add_edge(u, v)
                count += 1
                if count % 100000 == 0:
                    logging.debug('Loading...%d' % count)
        return nx.Graph(graph.subgraph(soc_nodes))

    def get_total_attr_graph(self, threshold=5):
        egos = []
        for root, dirs, files in os.walk(self.dirs['GPLUS']):
            for file in files:
                if file.endswith('.feat'):
                    egos.append(re.sub('.feat', '', file))
        total_attr_graph = nx.Graph()
        for ego in egos:
            if ego == '101560853443212199687':
                continue
            ego_net = GoogleEgo(ego)
            attr_graph = ego_net.prune_useless_part(threshold)
            logging.debug('[GPNetwork] #%s, true graph: %d, %d' % (ego,
                                                                   attr_graph.number_of_nodes(),
                                                                   attr_graph.number_of_edges()))

            total_attr_graph = nx.compose(total_attr_graph, attr_graph)
        logging.debug('[GPNetwork] total attribute graph (%d, %d) generated.' % (total_attr_graph.number_of_nodes(),
                                                                                 total_attr_graph.number_of_edges()))
        return total_attr_graph

    def get_nodes_attrs(self, attr_graph):
        soc_nodes = []
        attr_nodes = []
        for node in attr_graph.nodes():
            if node.isdigit():
                soc_nodes.append(node)
            else:
                attr_nodes.append(node)
        return soc_nodes, attr_nodes

    def write_attr_graph(self, threshold=5):
        total_attr_graph = self.get_total_attr_graph(threshold)
        nx.write_gexf(total_attr_graph, os.path.join(self.dirs['GPLUS'], 'total_attr.gexf'))
        logging.debug('[GPNetwork] total_attr.gexf written.')

    def write_soc_graph(self, attr_graph):
        soc_nodes, attr_nodes = self.get_nodes_attrs(attr_graph)
        total_soc_graph = self.get_total_soc_graph(soc_nodes)
        nx.write_gexf(total_soc_graph, os.path.join(self.dirs['GPLUS'], 'total_soc.gexf'))
        logging.debug('[GPNetwork] total_soc.gexf written.')

    def read_attr_graph(self, filename='total_attr.gexf'):
        t0 = time.time()
        graph = nx.read_gexf(os.path.join(self.dirs['GPLUS'], filename))
        logging.debug('[GPNetwork] total attr graph loaded in %f s' % (time.time() - t0))
        return graph

    def read_soc_graph(self, filename='total_soc.gexf'):
        t0 = time.time()
        graph = nx.read_gexf(os.path.join(self.dirs['GPLUS'], filename))
        logging.debug('[GPNetwork] total soc graph loaded in %f s' % (time.time() - t0))
        return graph

    def __init__(self):
        self.dirs = load_ranfig()
        self.soc_net = self.read_soc_graph()
        self.attr_net = self.read_attr_graph()
        self.soc_node, self.attr_node = self.get_nodes_attrs(self.attr_net)
        self.attr_edge = self.attr_net.edges()
        self.soc_edge = self.soc_net.edges()
        t0 = time.time()
        self.rpg = RPGraph(self.soc_node, self.attr_node, self.soc_edge, self.attr_edge, True)
        logging.debug('[GPNetwork] RPGraph Init. in %f s' % (time.time() - t0))

def test_code():
    egonet = GoogleEgo('104076158580173410325')
    graph = egonet.prune_useless_part(3)
    print(graph.number_of_nodes(), graph.number_of_edges())


def attr_lab_317():
    a = GooglePlusNetwork()
    rate = 1.0
    expr_settings = {
        'inst_google': rate,
        'job_manager': rate,
        'place_newyork': rate,
    }
    output_dir = "/Users/jiayichen/ranproject/res317-google/"
    expr = AttributeExperiment(a.rpg, expr_settings)
    utility, result_table = expr.delta_experiment(0.5, np.arange(0, 0.31, 0.03), 'equal')
    utility.to_csv(os.path.join(output_dir, 'utility.csv'))
    # expr.save_result_table(result_table, np.arange(0, 0.21, 0.02), output_dir)

def test_code2():
    a = GooglePlusNetwork()
    rate = 0.5
    expr_settings = {
        'inst_google': rate,
        'job_manager': rate,
        'place_newyork': rate,
    }
    expr = RelationExperiment(a.rpg, expr_settings)
    secrets, _ = expr.resampling()
    price = expr.auto_edge_price()
    a.rpg.random_directed(secrets, 0.5, 0.1)
    a.rpg.naive_bayes_directed(secrets, 0.5, 0.1, factor=0.5)
    a.rpg.entropy_directed(secrets, price, 0.5, 0.1)
    a.rpg.eppd_directed(secrets, price, 0.5, 0.1)


def relation_lab_0323():
    a = GooglePlusNetwork()
    rate = 0.5
    expr_settings = {
        'inst_google': rate,
        'job_manager': rate,
        'place_newyork': rate,
    }
    expr = RelationExperiment(a.rpg, expr_settings)
    output_dir = "/Users/jiayichen/ranproject/res324/"
    utility = expr.delta_directed(0.5, np.arange(0, 0.31, 0.03), rate, utility_name='Jaccard')
    utility.to_csv(os.path.join(output_dir, 'utility-J.csv'))

if __name__ == '__main__':
    # attr_lab_317()
    # test_code2()
    relation_lab_0323()
