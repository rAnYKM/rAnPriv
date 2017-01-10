# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
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


DEFAULT_FILENAME = 'facebook_complete'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FacebookNetwork:
    def attribute_stat(self):
        new_df_list = []
        for index, row in self.feat_table.iterrows():
            attr_name = row['attr']
            new_df_list.append({'attr': attr_name,
                                'count': len(self.ran.soc_attr_net.neighbors(attr_name))})
        attr_stat = pd.DataFrame(new_df_list)
        print attr_stat.sort_values('count', ascending=False)


    def __to_ran(self):
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
        self.ran = self.__to_ran()
        logging.debug('[snap_fbcomplete] Init Fin. in %f sec' % (time.time() - t0))

def main():
    a = FacebookNetwork()
    # a.attribute_stat()
    price = dict()
    secrets =dict()
    epsilon = dict()
    for i in a.ran.attr_node:
        price[i] = 1
    for n in a.ran.soc_node:
        if a.ran.soc_attr_net.has_edge(n, 'aenslid-538'):
            secrets[n] = ['aenslid-538']
            epsilon[n] = [0.4]
        else:
            secrets[n] = []
            epsilon[n] = []
    a.ran.d_knapsack_mask(secrets, price, epsilon)
    a.ran.s_knapsack_mask(secrets, price, epsilon, mode='greedy')

if __name__ == '__main__':
    main()
