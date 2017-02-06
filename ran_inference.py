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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


class InferenceAttack:

    def raw_attr_vector(self, secret):
        feature_x = np.array([[int(self.rpg.attr_net.has_edge(node, attr) and attr not in self.secrets[node])
                               for attr in self.rpg.attr_node if attr != secret]
                             for node in self.rpg.soc_node])
        return feature_x

    def get_labels(self, secret):
        label_y = np.array([int(self.rpg.attr_net.has_edge(node, secret))
                            for node in self.rpg.soc_node])
        return label_y

    def feature_sel(self, secret):
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        f_selector = SelectKBest(chi2, k=80)
        f_selector.fit(x, y)
        return f_selector

    def nb_classifier(self, secret):
        clf = BernoulliNB()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        fsl = self.feature_sel(secret)
        new_x = fsl.transform(x)
        clf.fit(new_x, y)
        new_y = clf.predict(new_x)
        return clf, fsl, self.evaluate(new_y, y)

    def svm_classifier(self, secret):
        clf = SVC()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        fsl = self.feature_sel(secret)
        new_x = fsl.transform(x)
        clf.fit(new_x, y)
        new_y = clf.predict(new_x)
        return clf, fsl, self.evaluate(new_y, y)

    def dt_classifier(self, secret):
        clf = DecisionTreeClassifier(random_state=0)
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        fsl = self.feature_sel(secret)
        new_x = fsl.transform(x)
        clf.fit(new_x, y)
        new_y = clf.predict(new_x)
        return clf, fsl, self.evaluate(new_y, y)

    def lr_classifier(self, secret):
        clf = LogisticRegression()
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        fsl = self.feature_sel(secret)
        new_x = fsl.transform(x)
        clf.fit(new_x, y)
        new_y = clf.predict(new_x)
        return clf, fsl, self.evaluate(new_y, y)

    def evaluate(self, new_y, y):
        return precision_score(y, new_y), recall_score(y, new_y), f1_score(y, new_y)

    def score(self, clf, secret):
        x = self.raw_attr_vector(secret)
        y = self.get_labels(secret)
        return cross_val_score(clf, x, y, scoring='f1', cv=3)


    def __init__(self, rpgraph, secrets):
        self.rpg = rpgraph
        self.secrets = secrets


def rpg_attr_vector(rpg, secret, secrets):
    feature_x = np.array([[int(rpg.attr_net.has_edge(node, attr) and attr not in secrets[node])
                           for attr in rpg.attr_node if attr != secret]
                          for node in rpg.soc_node])
    return feature_x


def rpg_labels(rpg, secret):
    label_y = np.array([int(rpg.attr_net.has_edge(node, secret))
                        for node in rpg.soc_node])
    return label_y


def infer_performance(clf, fsl, t_x, t_y):
    t_x_new = fsl.transform(t_x)
    t_y_new = clf.predict(t_x_new)
    return precision_score(t_y, t_y_new), recall_score(t_y, t_y_new), f1_score(t_y, t_y_new), \
           accuracy_score(t_y, t_y_new)
