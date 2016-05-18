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

    def __init__(self, ego_id):
        self.dir = load_ranfig()
        self.root = ego_id
