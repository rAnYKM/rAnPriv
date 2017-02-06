# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _
#    _____/   |  ____  / __ \_____(_)   __
#   / ___/ /| | / __ \/ /_/ / ___/ / | / /
#  / /  / ___ |/ / / / ____/ /  / /| |/ /
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/
#
# Script Name: snap_fbcomplete.py
# Date: Dec. 21, 2016

import os
import logging
import time
import pandas as pd
import networkx as nx
from ranfig import load_ranfig
from ran_graph import RanGraph
from ran_priv import RPGraph
from ran_inference import InferenceAttack, infer_performance, rpg_labels, rpg_attr_vector


DEFAULT_FILENAME = 'facebook_complete'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FacebookNetwork:
    def attribute_stat(self):
        new_df_list = []
        for index, row in self.feat_table.iterrows():
            attr_name = row['attr']
            new_df_list.append({'attr': attr_name,
                                'count': len(self.rpg.attr_net.neighbors(attr_name))})
        attr_stat = pd.DataFrame(new_df_list)
        print(attr_stat.sort_values('count', ascending=False))

    def __to_ran(self):
        """ Import the SNAP Facebook dataset as a RanGraph """
        # soc_node, attr_node, soc_edge, attr_edge
        soc_node = self.network.nodes()
        soc_edge = self.network.edges()
        # get attributes
        attr_node = self.feat_table['attr'].tolist()
        # get attribute links
        attr_edge = list()
        for index, row in self.node_table.iterrows():
            uid = str(row['user_id'])
            profile = str(row['profile'])
            links = [(uid, self.feat_table['attr'].iloc[int(ind)]) for ind in profile.split(' ')
                     if ind != 'nan']
            attr_edge += links
        ran = RanGraph(soc_node, attr_node, soc_edge, attr_edge, False)
        return ran

    def __to_rpg(self):
        """ Import the SNAP Facebook dataset as a RPGraph """
        # soc_node, attr_node, soc_edge, attr_edge
        soc_node = self.network.nodes()
        soc_edge = self.network.edges()
        # get attributes
        attr_node = self.feat_table['attr'].tolist()
        # get attribute links
        attr_edge = list()
        for index, row in self.node_table.iterrows():
            uid = str(row['user_id'])
            profile = str(row['profile'])
            links = [(uid, self.feat_table['attr'].iloc[int(ind)]) for ind in profile.split(' ')
                     if ind != 'nan']
            attr_edge += links
        rpg = RPGraph(soc_node, attr_node, soc_edge, attr_edge, False)
        return rpg

    def __init__(self, filename=DEFAULT_FILENAME):
        t0 = time.time()
        self.dirs = load_ranfig()
        # Load Facebook Completed Network
        self.network = nx.Graph()
        self.network = nx.read_graphml(os.path.join(self.dirs['FBOOK'], filename + '.graphml'))
        logging.debug('Facebook Complete loaded with %d nodes and %d edges'
                      % (self.network.number_of_nodes(), self.network.number_of_edges()))
        # Load Node Features
        self.feat_table = pd.read_csv(os.path.join(self.dirs['FBOOK'], filename + '.feats'))
        self.node_table = pd.read_csv(os.path.join(self.dirs['FBOOK'], filename + '.nodes'))
        # self.ran = self.__to_ran()
        self.rpg = self.__to_rpg()
        logging.debug('[snap_fbcomplete] Init Fin. in %f sec' % (time.time() - t0))

def main():
    a = FacebookNetwork()
    # a.attribute_stat()
    price = dict()
    rprice = dict()
    secrets =dict()
    for i in a.rpg.attr_node:
        price[i] = 1
    secret = 'aenslid-538'
    for n in a.rpg.soc_node:
        if a.rpg.attr_net.has_edge(n, secret):
            secrets[n] = [secret]
        else:
            secrets[n] = []
    print(a.rpg.affected_attribute_number(secrets))
    epsilon = 0.1
    delta = 0.4

    org = InferenceAttack(a.rpg, secrets)
    clf, fsl, result = org.dt_classifier(secret)
    score = org.score(clf, secret)
    print(result, score, infer_performance(clf, fsl, rpg_attr_vector(a.rpg, secret, secrets), rpg_labels(a.rpg, secret)))
    t0 = time.time()
    new_ran = a.rpg.d_knapsack_mask(secrets, price, epsilon, delta, mode='greedy')
    print(time.time() - t0)
    print(a.rpg.cmp_attr_degree_L1_error(new_ran))
    def1 = InferenceAttack(new_ran, secrets)
    clf2, fsl2, result25 = def1.dt_classifier(secret)
    print(result25, def1.score(clf2, secret),
          infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret)))
    """
    t0 = time.time()
    a.rpg.naive_bayes_mask(secrets, epsilon, delta, 0.1)
    print(time.time() - t0)
    t0 = time.time()
    a.rpg.entropy_mask(secrets, epsilon, delta)
    print(time.time() - t0)
    """
    t0 = time.time()
    new_ran = a.rpg.v_knapsack_mask(secrets, price, epsilon, delta, mode='greedy')
    # weight = {n: [a.rpg.get_max_weight(secret, epsilon, delta)] for n in a.ran.soc_net.nodes()}
    # old_ran = a.ran.s_knapsack_mask(secrets, price, weight, mode='greedy')
    print(time.time() - t0)
    print(a.rpg.cmp_attr_degree_L1_error(new_ran))
    def2 = InferenceAttack(new_ran, secrets)
    clf3, fsl3, result35 = def1.dt_classifier(secret)
    print(result35, def2.score(clf3, secret),
          infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret)))
    for i in a.rpg.soc_net.edges():
        rprice[i] = 1
    # t0 = time.time()
    # a.ran.s_knapsack_relation_global(secrets, rprice, epsilon)
    # print(time.time() - t0)
    # print('3734' in a.rpg.neighbor_array)
    '''
    t0 = time.time()
    new_ran = a.rpg.d_knapsack_relation(secrets, rprice, epsilon, delta)
    print(time.time() - t0)
    print(a.rpg.cmp_soc_degree_L1_error(new_ran))
    t0 = time.time()
    new_ran = a.rpg.v_knapsack_relation(secrets, rprice, epsilon, delta)
    print(time.time() - t0)
    print(a.rpg.cmp_soc_degree_L1_error(new_ran))
    '''

if __name__ == '__main__':
    main()

