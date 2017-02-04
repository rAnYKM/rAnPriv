# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _
#    _____/   |  ____  / __ \_____(_)   __
#   / ___/ /| | / __ \/ /_/ / ___/ / | / /
#  / /  / ___ |/ / / / ____/ /  / /| |/ /
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/
#
# Script Name: ran_inference.py
# Date: Jan. 11, 2017

import numpy as np
from ran_priv import RPGraph
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

class InferenceAttack:
    def rpg_attr_vector(self, rpg, secret):
        feature_x = np.array([[int(rpg.attr_net.has_edge(node, attr) and attr not in self.secrets[node])
                               for attr in rpg.attr_node if attr != secret]
                              for node in rpg.soc_node])
        return feature_x

    def rpg_labels(self, rpg, secret):
        label_y = np.array([int(rpg.attr_net.has_edge(node, secret))
                            for node in rpg.soc_node])
        return label_y

    def raw_attr_vector(self, secret):
        feature_x = np.array([[int(self.rpg.attr_net.has_edge(node, attr) and attr not in self.secrets[node])
                               for attr in self.rpg.attr_node if attr != secret]
                             for node in self.rpg.soc_node])
        return feature_x

    def get_labels(self, secret):
        label_y = np.array([int(self.rpg.attr_net.has_edge(node, secret))
                            for node in self.rpg.soc_node])
        return label_y

    def nb_classifier(self, secret):
        clf = BernoulliNB()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        clf.fit(x, y)
        new_y = clf.predict(x)
        return clf, self.evaluate(new_y, y)

    def svm_classifier(self, secret):
        clf = SVC()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        clf.fit(x, y)
        new_y = clf.predict(x)
        return clf, self.evaluate(new_y, y)

    def dt_classifier(self, secret):
        clf = DecisionTreeClassifier(random_state=0)
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        clf.fit(x, y)
        new_y = clf.predict(x)
        return clf, self.evaluate(new_y, y)

    def lr_classifier(self, secret):
        clf = LogisticRegression()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        clf.fit(x, y)
        new_y = clf.predict(x)
        return clf, self.evaluate(new_y, y)

    def evaluate(self, new_y, y):
        return precision_score(y, new_y), recall_score(y, new_y), f1_score(y, new_y)

    def __init__(self, rpgraph, secrets):
        self.rpg = rpgraph
        self.secrets = secrets
