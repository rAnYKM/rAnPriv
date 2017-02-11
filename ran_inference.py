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

import os
import csv
import numpy as np
from ran_priv import RPGraph
from ranfig import load_ranfig
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
        return cross_val_score(clf, x, y, scoring='f1', cv=5)

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
    return precision_score(t_y, t_y_new), recall_score(t_y, t_y_new), f1_score(t_y, t_y_new)


class RelationAttack:
    # The class is used to generate the data for NetKit
    # For more information, please visit http://netkit-srl.sourceforge.net/
    def generate_data_set(self, secret):
        soc_node = self.rpg.soc_node
        attr_node = self.rpg.attr_node
        soc_edge = self.rpg.soc_edge
        attr_edge = self.rpg.attr_edge
        yes_no = {True: 'yes', False: 'no'}
        # write soc nodes
        soc2write = [{'UID': node,
                      'Secret': yes_no[self.rpg.attr_net.has_edge(node, secret)]}
                     for node in soc_node]
        with open(os.path.join(self.path, self.filename + '-soc.csv'), 'w') as fp:
            writer = csv.DictWriter(fp, ['UID', 'Secret'])
            for i in soc2write:
                writer.writerow(i)

        # write attr nodes
        attr2write = [{'AID': node}
                      for node in attr_node]
        with open(os.path.join(self.path, self.filename + '-attr.csv'), 'w') as fp:
            writer = csv.DictWriter(fp, ['AID'])
            for i in attr2write:
                writer.writerow(i)

        # write soc edges
        edge2write = [{'Source': e[0], 'Destination': e[1], 'Weight': 1}
                      for e in soc_edge]
        with open(os.path.join(self.path, self.filename + '-edge.rn'), 'w') as fp:
            writer = csv.DictWriter(fp, ['Source', 'Destination', 'Weight'])
            for i in edge2write:
                writer.writerow(i)

        # write attr edges
        link2write = [{'Source': e[0], 'Destination': e[1], 'Weight': 1}
                      for e in attr_edge
                      if e[1] not in self.secrets[e[0]]]
        with open(os.path.join(self.path, self.filename + '-link.rn'), 'w') as fp:
            writer = csv.DictWriter(fp, ['Source', 'Destination', 'Weight'])
            for i in link2write:
                writer.writerow(i)

        schema = """# This is a NetKit schema generated by ran_inference.py
@nodetype SocialNode
@attribute UID key
@attribute Secret CATEGORICAL
@nodedata """ + self.filename + '-soc.csv' + \
                 """
@nodetype AttributeNode
@attribute AID key
@nodedata """ + self.filename + '-attr.csv' + \
                 """
@edgetype Linked SocialNode SocialNode
@Reversible
@edgedata """ + self.filename + '-edge.rn' + \
                 """
@edgetype Has SocialNode AttributeNode
@edgedata """ + self.filename + '-link.rn'
        with open(os.path.join(self.path, self.filename + '.arff'), 'w') as fp:
            fp.write(schema)

    def __init__(self, rpg, secrets, filename='webkit'):
        # Schema Template
        self.rpg = rpg
        self.secrets = secrets
        self.filename = filename
        tmp_dict = load_ranfig()
        self.path = tmp_dict['OUT']
