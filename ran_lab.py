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

import time
import logging
import numpy as np
from ran_priv import RPGraph
from snap_fbcomplete import FacebookNetwork
from ran_inference import InferenceAttack, infer_performance, rpg_attr_vector, rpg_labels


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def single_attribute_test(secret, epsilon, delta):
    a = FacebookNetwork()
    price = dict()
    rprice = dict()
    secrets = dict()
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
    ## 2 kinds of test methods:
    ## 1. (O)All, (A)All (all data for train, all data for test)
    ## 2. (A)k-Fold (k-1 for train, 1 for test)

    logging.info('[ran_lab] Origin %d-fold (f1 score) - average=%f'
                  % (len(score), np.average(score)))
    logging.info('[ran_lab] Origin full graph - precision=%f, recall=%f, f1-score=%f'
                  % (full_att[0], full_att[1], full_att[2]))
    t0 = time.time()
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

    new_ran = a.rpg.v_knapsack_mask(secrets, price, epsilon, delta, mode='greedy')
    # weight = {n: [a.rpg.get_max_weight(secret, epsilon, delta)] for n in a.ran.soc_net.nodes()}
    # old_ran = a.ran.s_knapsack_mask(secrets, price, weight, mode='greedy')
    print(time.time() - t0)
    print(a.rpg.cmp_attr_degree_L1_error(new_ran))
    def2 = InferenceAttack(new_ran, secrets)
    clf3, fsl3, result35 = def1.dt_classifier(secret)
    score = def2.score(clf3, secret)
    full_att = infer_performance(clf, fsl, rpg_attr_vector(new_ran, secret, secrets), rpg_labels(new_ran, secret))
    logging.info('[ran_lab] Origin %d-fold (f1 score) - average=%f'
                  % (len(score), np.average(score)))
    logging.info('[ran_lab] Origin full graph - precision=%f, recall=%f, f1-score=%f'
                  % (full_att[0], full_att[1], full_att[2]))

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
    single_attribute_test('aenslid-538', 0.1, 0)
