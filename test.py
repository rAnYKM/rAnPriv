# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: test.py
# Date: July. 5, 2016

import numpy as np
import logging
from snap_facebook import FacebookEgoNet


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def experiment(data, secret_cate, ar, dr, e):
    fb = FacebookEgoNet(data)
    # Build secret dict
    secrets = dict()
    epsilon = dict()
    ep2 = dict()
    for node in fb.ran.soc_net.nodes():
        feature = [attr for attr in fb.ran.soc_attr_net.neighbors(node)
                   if attr[0] == 'a']
        # secret = [attr for attr in feature if attr.split('-')[0] in secret_cate]
        secret = [attr for attr in feature if attr in secret_cate]
        secrets[node] = secret
        epsilon[node] = [np.log2(e) -
                         np.log2(len(fb.ran.soc_attr_net.neighbors(a))/float(fb.ran.soc_net.number_of_nodes()))
                         for a in secret]
        ep2[node] = [e] * len(secret)
    price = fb.ran.value_of_attribute('equal')
    att_ran = fb.ran.random_sampling(ar)

    score_li = []
    stat_li = []
    res_li = []
    for times in range(0,10,1):
        def_ran, stat01, _ = fb.ran.adv_random_masking(secrets, ep2)
        _, res02 = def_ran.inference_attack(secrets, att_ran)
        _, score01 = def_ran.utility_measure(secrets, price)
        score_li.append(score01)
        stat_li.append(stat01)
        res_li.append(res02)
    stat1 = np.average(stat_li)
    score1 = np.average(score_li)
    res2 = np.average(res_li)

    greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon)
    ddp, stat5 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'dp')
    s_good, stat3 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp')
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy')
    _, res = fb.ran.inference_attack(secrets, att_ran)

    _, res5 = greedy.inference_attack(secrets, att_ran)
    _, res6 = ddp.inference_attack(secrets, att_ran)
    _, res7 = s_good.inference_attack(secrets, att_ran, e)
    _, res8 = s_greedy.inference_attack(secrets, att_ran, e)

    _, max_score = fb.ran.utility_measure(secrets, price)

    _, score2 = greedy.utility_measure(secrets, price)
    _, score5 = ddp.utility_measure(secrets, price)
    _, score3 = s_good.utility_measure(secrets, price)
    _, score4 = s_greedy.utility_measure(secrets, price)
    # _, score5 = s_dual.utility_measure(secrets, price)
    # d, res10 = s_gual.inference_attack(secrets, fb.ran, e)
    print res, res2, res5, res6, res7, res8
    # print a, b
    """
    for i in range(len(tp1)):
        if tp1[i][0] != tp2[i][0]:
            print tp1[i], tp2[i]
    """
    stat = [stat1, stat2, stat5, stat3, stat4]
    ress = [res2, res5, res6, res7, res8]
    scos = [score1, score2, score5, score3, score4]
    n_scos = [i/float(max_score) for i in scos]
    reff = res
    return stat, ress, reff, n_scos


def experiment_relation(data, secret_cate, ar, dr, e):
    fb = FacebookEgoNet(data)
    # Build secret dict
    secrets = dict()
    epsilon = dict()
    ep2 = dict()
    for node in fb.ran.soc_net.nodes():
        feature = [attr for attr in fb.ran.soc_attr_net.neighbors(node)
                   if attr[0] == 'a']
        # secret = [attr for attr in feature if attr.split('-')[0] in secret_cate]
        secret = [attr for attr in feature if attr in secret_cate]
        secrets[node] = secret
        epsilon[node] = [np.log2(e) -
                         np.log2(len(fb.ran.soc_attr_net.neighbors(a))/float(fb.ran.soc_net.number_of_nodes()))
                         for a in secret]
        ep2[node] = [e] * len(secret)
    price = fb.ran.value_of_attribute('common')
    price2 = fb.ran.value_of_relation('equal')
    price3 = fb.ran.value_of_edge('equal')
    att_ran = fb.ran.random_sampling(ar)
    def_ran, stat1, _ = fb.ran.adv_random_masking(secrets, ep2)
    greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon)
    greedy2, stat3 = greedy.d_knapsack_relation(secrets, price2, epsilon)
    s_greedy, stat4 = fb.ran.s_knapsack_relation_global(secrets, price3, ep2)
    _, res4 = def_ran.inference_attack_relation(secrets, att_ran)
    _, res3 = fb.ran.inference_attack_relation(secrets, att_ran)
    _, res6 = greedy.inference_attack_relation(secrets, att_ran)
    _, res8 = s_greedy.inference_attack_relation(secrets, att_ran)
    print res3, res4, res6, res8
    """
    for i in range(len(tp1)):
        if tp1[i][0] != tp2[i][0]:
            print tp1[i], tp2[i]
    """
    stat = [stat1, stat3, stat4]
    ress = [res4, res6, res8]
    reff = res3
    return stat, ress, reff

def data_record(xs, ys, filename):
    with open(filename, 'w') as fp:
        for x in xs:
            fp.write(' '.join([str(i) for i in x]) + '\n')
        fp.write('\n')
        for y in ys:
            fp.write(' '.join([str(i) for i in y]) + '\n')

if __name__ == '__main__':
    sec = ['aensl-50']
    s1 = []
    s2 = []
    s3 = []
    for i in np.arange(0.4, 1, 0.05):
        print i
        stat, ress, reff, socs = experiment('0', sec, 1, 1, i)
        s1.append(stat)
        s2.append(ress)
        # s3.append(scos)
    data_record([np.arange(0.05, 1, 0.05)], s1, 'out/exp_0_attr.txt')
    # data_record([np.arange(0.05, 1, 0.05)], s2, 'performance3.txt')
    # data_record([np.arange(0.05, 1, 0.05)], s3, 'score3.txt')
