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
import pandas as pd
import networkx as nx
from collections import Counter
from ranfig import load_ranfig
from ran_graph import RanGraph


DEFAULT_FILENAME = 'facebook_complete'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FacebookNetwork:
    def __init__(self, filename=DEFAULT_FILENAME):
        self.dirs = load_ranfig()
        # Load Facebook Completed Network
        self.network = nx.Graph()
        self.network = nx.read_graphml(os.path.join(self.dirs['FBOOK'], filename + '.graphml'))
        logging.debug('Facebook Complete loaded with %d nodes and %d edges'
                      % (self.network.number_of_nodes(), self.network.number_of_edges()))

def main():
    a = FacebookNetwork()

if __name__ == '__main__':
    main()


