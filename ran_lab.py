# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _
#    _____/   |  ____  / __ \_____(_)   __
#   / ___/ /| | / __ \/ /_/ / ___/ / | / /
#  / /  / ___ |/ / / / ____/ /  / /| |/ /
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/
#
# Script Name: ran_lab.py
# Date: Feb. 5, 2017

from __future__ import division
import logging
import numpy as np
import pandas as pd
from snap_fbcomplete import FacebookNetwork
from ran_inference import InferenceAttack, infer_performance, rpg_attr_vector, rpg_labels, RelationAttack

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def single_attribute_test(secret, epsilon, delta):
    a = FacebookNetwork()
    price = dict()
    rprice = dict()
    secrets = dict()
    exp1 = dict()
    exp2 = dict()
    for i in a.rpg.attr_node:
        price[i] = 1
    for n in a.rpg.soc_node:
        if a.rpg.attr_net.has_edge(n, secret):
            secrets[n] = [secret]
        else:
            secrets[n] = []
    # Basic Info Display
    logging.debug('[ran_lab] Single Attribute Test - e=%f, d=%f, threshold=%f, affected attributes=%d'
                  % (epsilon,
                     delta,
                     a.rpg.get_max_weight(secret, epsilon, delta),
                     a.rpg.affected_attribute_number(secrets)))
    # Original Graph
    org = InferenceAttack(a.rpg, secrets)
    clf, fsl, result = org.dt_classifier(secret)
    score = org.score(clf, secret)
    full_att = infer_performance(clf, fsl, rpg_attr_vector(a.rpg, secret, secrets), rpg_labels(a.rpg, secret))
    exp1['origin'] = np.average(score)
    exp2['origin'] = full_att[2]
    logging.info('[ran_lab] Origin %d-fold (f1 score) - average=%f'
                 % (len(score), np.average(score)))
    logging.info('[ran_lab] Origin full graph - precision=%f, recall=%f, f1-score=%f'
                 % (full_att[0], full_att[1], full_att[2]))
    """"
    new_ran = a.rpg.d_knapsack_mask(secrets, price, epsilon, delta, mode='greedy')
    print(time.time() - t0)
    print(a.rpg.cmp_attr_degree_L1_error(new_ran))
    def1 = InferenceAttack(new_ran, secrets)
    clf2, fsl2, result25 = def1.dt_classifier(secret)
    print(result25, def1.score(clf2, secret),
          infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret)))
    """
    # a.rpg.naive_bayes_mask(secrets, epsilon, delta, 0.1)

    new_ran = a.rpg.entropy_mask(secrets, epsilon, delta)
    def1 = InferenceAttack(new_ran, secrets)
    clf2, fsl2, result25 = def1.dt_classifier(secret)
    score = def1.score(clf2, secret)
    full_att = infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret))
    logging.info('[ran_lab] Origin %d-fold (f1 score) - average=%f'
                 % (len(score), np.average(score)))
    logging.info('[ran_lab] Origin full graph - precision=%f, recall=%f, f1-score=%f'
                 % (full_att[0], full_att[1], full_att[2]))

    exp1['entropy'] = np.average(score)
    exp2['entropy'] = full_att[2]
    new_ran = a.rpg.v_knapsack_mask(secrets, price, epsilon, delta, mode='greedy')
    # weight = {n: [a.rpg.get_max_weight(secret, epsilon, delta)] for n in a.ran.soc_net.nodes()}
    # old_ran = a.ran.s_knapsack_mask(secrets, price, weight, mode='greedy')
    # print(time.time() - t0)
    # print(a.rpg.cmp_attr_degree_L1_error(new_ran))
    def2 = InferenceAttack(new_ran, secrets)
    clf3, fsl3, result35 = def1.dt_classifier(secret)
    score = def2.score(clf3, secret)
    full_att = infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret))
    logging.info('[ran_lab] Origin %d-fold (f1 score) - average=%f'
                 % (len(score), np.average(score)))
    logging.info('[ran_lab] Origin full graph - precision=%f, recall=%f, f1-score=%f'
                 % (full_att[0], full_att[1], full_att[2]))
    exp1['vkp'] = np.average(score)
    exp2['vkp'] = full_att[2]
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
    return exp1, exp2


class AttackSimulator:
    def self_attack(self):
        org = InferenceAttack(self.rpg, self.secrets)
        clf, fsl, result = org.dt_classifier(self.secret)
        # score = org.score(clf, secret)
        full_att = infer_performance(clf,
                                     fsl,
                                     rpg_attr_vector(self.rpg, self.secret, self.secrets),
                                     rpg_labels(self.rpg, self.secret))
        return {'classifier': clf,
                'feat_selector': fsl,
                'score': full_att}

    def rpg_attack(self, rpg):
        full_att = infer_performance(self.clf,
                                     self.fsl,
                                     rpg_attr_vector(rpg, self.secret, self.secrets),
                                     rpg_labels(rpg, self.secret))
        logging.info('[ran_lab] Full Graph Attack - precision=%f, recall=%f, f1-score=%f'
                     % (full_att[0], full_att[1], full_att[2]))
        # TODO: Utility Measurement
        total = len(self.rpg.attr_edge)
        remain = len(rpg.attr_edge)
        secs = total - len(self.rpg.attr_net.neighbors(self.secret))
        utility = (total - remain) / secs
        return {'score': full_att,
                'utility': 1 - utility}

    def config(self, secret, epsilon, delta):
        self.secret = secret
        self.epsilon = epsilon
        self.delta = delta
        org_settings = self.self_attack()
        self.clf = org_settings['classifier']
        self.fsl = org_settings['feat_selector']
        self.score = org_settings['score']

    def __init__(self, rpg, secrets, secret, epsilon=0.1, delta=0.0):
        self.rpg = rpg
        self.secrets = secrets
        self.secret = secret
        self.epsilon = epsilon
        self.delta = delta
        org_settings = self.self_attack()
        self.clf = org_settings['classifier']
        self.fsl = org_settings['feat_selector']
        self.score = org_settings['score']


def single_attack_test_ver2(simulator, price, secret, epsilon, delta):
    simulator.config(secret, epsilon, delta)
    # Entropy Masking
    new_ran = simulator.rpg.entropy_mask(simulator.secrets, epsilon, delta)
    result = simulator.rpg_attack(new_ran)
    etp_res = {'precision': result['score'][0],
               'recall': result['score'][1],
               'f1': result['score'][2],
               'utility': result['utility']}

    # VKP Masking
    new_ran = simulator.rpg.v_knapsack_mask(simulator.secrets, price, epsilon, delta, mode='greedy')
    result = simulator.rpg_attack(new_ran)
    vkp_res = {'precision': result['score'][0],
               'recall': result['score'][1],
               'f1': result['score'][2],
               'utility': result['utility']}
    return etp_res, vkp_res


def single_attribute_batch(secret, epsilon, delta_range):
    exp1 = list()
    exp2 = list()
    for delta in delta_range:
        e1, e2 = single_attribute_test(secret, epsilon, delta)
        exp1.append(e1)
        exp2.append(e2)
    df1 = pd.DataFrame(exp1, index=delta_range)
    df2 = pd.DataFrame(exp2, index=delta_range)
    df1.to_csv('out/%s-exp1.csv' % secret)
    df2.to_csv('out/%s-exp2.csv' % secret)


def single_attribute_batch_ver2(secret, epsilon, delta_range):
    a = FacebookNetwork()
    price = dict()
    secrets = dict()
    for i in a.rpg.attr_node:
        price[i] = 1
    for n in a.rpg.soc_node:
        if a.rpg.attr_net.has_edge(n, secret):
            secrets[n] = [secret]
        else:
            secrets[n] = []
    # Basic Info Display
    logging.debug('[ran_lab] Single Attribute Test - affected attributes=%d'
                  % (a.rpg.affected_attribute_number(secrets)))
    rpg = a.rpg
    # performance - utility
    simulator = AttackSimulator(rpg, secrets, secret)
    res1 = []
    res2 = []
    for delta in delta_range:
        etp_res, vkp_res = single_attack_test_ver2(simulator, price, secret, epsilon, delta)
        res1.append(etp_res)
        res2.append(vkp_res)
    df1 = pd.DataFrame(res1, index=delta_range)
    df2 = pd.DataFrame(res2, index=delta_range)
    df1.to_csv('out/%s-res1.csv' % secret)
    df2.to_csv('out/%s-res2.csv' % secret)


class RelationAttackSimulator:
    def attack(self, sample_rate):
        result = self.attacker.cross_validation(10, sample_rate)
        return self.attacker.result_formatter(result, self.secret)

    def config(self, secret, epsilon, delta):
        self.secret = secret
        self.epsilon = epsilon
        self.delta = delta
        self.attacker.generate_data_set(secret)

    def __init__(self, rpg, secrets, secret, filename, epsilon=0.1, delta=0.0):
        self.rpg = rpg
        self.secrets = secrets
        self.secret = secret
        self.epsilon = epsilon
        self.delta = delta
        self.filename = filename
        self.attacker = RelationAttack(self.rpg, self.secrets, self.filename)
        self.attacker.generate_data_set(secret)


def tmp_relation_test():
    a = FacebookNetwork()
    price = dict()
    rprice = dict()
    secrets = dict()
    secret = 'aenslid-52'
    epsilon = 0.1
    delta = 0.01
    for i in a.rpg.attr_node:
        price[i] = 1
    for n in a.rpg.soc_node:
        if a.rpg.attr_net.has_edge(n, secret):
            secrets[n] = [secret]
        else:
            secrets[n] = []
    simulator = RelationAttackSimulator(a.rpg, secrets, secret, 'origin', 0.1, 0.01)
    print(simulator.attack(0.6))
    for i in a.rpg.soc_net.edges():
        rprice[i] = 1
    new_ran = a.rpg.d_knapsack_relation(secrets, rprice, epsilon, delta)
    simulator = RelationAttackSimulator(new_ran, secrets, secret, 'dkp', 0.1, 0.01)
    print(simulator.attack(0.6))


if __name__ == '__main__':
    # single_attribute_test('aenslid-538', 0.1, 0)
    # single_attribute_batch_ver2('aenslid-52', 0.1, np.arange(0, 1.0, 0.1))
    tmp_relation_test()

