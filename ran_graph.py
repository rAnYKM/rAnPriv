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
# Date: May. 18, 2016


import networkx as nx


class RanGraph:
    """
    A Ran Social Network Graph Involves the following three sub-graphs:
    social network graph (social nodes, also actors)
    social attribute network graph (social nodes with attribute nodes)
    attribute network graph (attribute nodes)
    """

    @property
    def is_directed(self):
        return self._is_directed

    @is_directed.setter
    def is_directed(self, val):
        if val:
            self._is_directed = True
        else:
            self._is_directed = False

    def __init__(self, soc, soc_attr, attr, is_directed=False):
        self.is_directed = is_directed
        self.soc_graph = soc
        self.soc_attr_graph = soc_attr
        self.attr_graph = attr
