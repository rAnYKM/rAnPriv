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
    mode = 'single'
    score_li = []
    stat_li = []
    res_li = []
    fs_li = []
    for times in range(0,10,1):
        def_ran, stat01, _ = fb.ran.adv_random_masking(secrets, ep2)
        _, res01, fs01 = def_ran.inference_attack(secrets, att_ran, ep2)
        _, score01 = def_ran.utility_measure(secrets, price, mode)
        score_li.append(score01)
        stat_li.append(stat01)
        res_li.append(res01)
        fs_li.append(fs01)
    stat1 = np.average(stat_li)
    score1 = np.average(score_li)
    res1 = np.average(res_li)
    fs1 = np.average(fs_li)

    greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'greedy', mode)
    ddp, stat3 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'dp', mode)
    s_good, stat5 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp', mode)
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy', mode)
    _, res, fs = fb.ran.inference_attack(secrets, att_ran, ep2)
    _, res2, fs2 = greedy.inference_attack(secrets, att_ran, ep2)
    _, res3, fs3 = ddp.inference_attack(secrets, att_ran, ep2)
    _, res5, fs5 = s_good.inference_attack(secrets, att_ran, ep2)
    _, res4, fs4 = s_greedy.inference_attack(secrets, att_ran, ep2)

    _, max_score = fb.ran.utility_measure(secrets, price, mode)
    _, score2 = greedy.utility_measure(secrets, price, mode)
    _, score3 = ddp.utility_measure(secrets, price, mode)
    _, score5 = s_good.utility_measure(secrets, price, mode)
    _, score4 = s_greedy.utility_measure(secrets, price, mode)
    stat = [stat1, stat2, stat3, stat4, stat5]
    ress = [res1, res2, res3, res4, res5]
    scos = [score1, score2, score3, score4, score5]
    n_scos = [i / float(max_score) for i in scos]
    reff = res
    print fs, fs1, fs2, fs3, fs4, fs5
    fss = [fs2, fs3]
    return stat, ress, reff, n_scos, fss


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
    # price = fb.ran.value_of_attribute('common')
    price2 = fb.ran.value_of_relation('equal')
    price3 = fb.ran.value_of_edge('equal')
    att_ran = fb.ran.random_sampling(ar)
    def_ran, stat1, _ = fb.ran.adv_random_masking(secrets, ep2, 'on')
    # greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon)
    greedy2, stat3 = fb.ran.d_knapsack_relation(secrets, price2, epsilon)
    s_greedy, stat4 = fb.ran.s_knapsack_relation_global(secrets, price3, ep2)
    _, res4 = def_ran.inference_attack_relation(secrets, att_ran)
    _, res3 = fb.ran.inference_attack_relation(secrets, att_ran)
    _, res6 = greedy2.inference_attack_relation(secrets, att_ran)
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

def experiment_strict_point(data, secret_cate, ar, dr, e):
    fb = FacebookEgoNet(data)
    secrets = dict()
    epsilon = dict()
    ep2 = dict()
    for node in fb.ran.soc_net.nodes():
        feature = [attr for attr in fb.ran.soc_attr_net.neighbors(node)
                   if attr[0] == 'a']
        # secret = [attr for attr in feature if attr.split('-')[0] in secret_cate]
        secret = [attr for attr in feature if attr in secret_cate]
        secrets[node] = secret
        epsilon[node] = [0 for a in secret]
        ep2[node] = [len(fb.ran.soc_attr_net.neighbors(a))/float(fb.ran.soc_net.number_of_nodes())
                     for a in secret]
        print ep2[node],
    print '===end==='
    price = fb.ran.value_of_attribute('equal')
    att_ran = fb.ran.random_sampling(ar)
    score_li = []
    stat_li = []
    res_li = []
    fs_li = []
    for times in range(0, 10, 1):
        def_ran, stat01, _ = fb.ran.adv_random_masking(secrets, ep2)
        _, res01, fs01 = def_ran.inference_attack(secrets, att_ran, ep2)
        _, score01 = def_ran.utility_measure(secrets, price)
        score_li.append(score01)
        stat_li.append(stat01)
        res_li.append(res01)
        fs_li.append(fs01)
    stat1 = np.average(stat_li)
    score1 = np.average(score_li)
    res1 = np.average(res_li)
    fs1 = np.average(fs_li)

    greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon)
    ddp, stat3 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'dp')
    s_good, stat5 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp')
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy')
    _, res, fs = fb.ran.inference_attack(secrets, att_ran, ep2)
    _, res2, fs2 = greedy.inference_attack(secrets, att_ran, ep2)
    _, res3, fs3 = ddp.inference_attack(secrets, att_ran, ep2)
    _, res5, fs5 = s_good.inference_attack(secrets, att_ran, ep2)
    _, res4, fs4 = s_greedy.inference_attack(secrets, att_ran, ep2)

    _, max_score = fb.ran.utility_measure(secrets, price)
    _, score2 = greedy.utility_measure(secrets, price)
    _, score3 = ddp.utility_measure(secrets, price)
    _, score5 = s_good.utility_measure(secrets, price)
    _, score4 = s_greedy.utility_measure(secrets, price)
    stat = [stat1, stat2, stat3, stat4, stat5]
    ress = [res1, res2, res3, res4, res5]
    scos = [score1, score2, score3, score4, score5]
    n_scos = [i / float(max_score) for i in scos]
    reff = res
    print fs, fs1, fs2, fs3, fs4, fs5
    return stat, ress, reff, n_scos

def shell_bar():
    sec = ['aensl-538']
    s1 = []
    s2 = []
    s3 = []
    stat, ress, reff, scos = experiment_strict_point('1684', sec, 1, 1, 0.35)
    s1.append(stat)
    s2.append(ress)
    s3.append(scos)
    # print stat, ress, reff, scos
    print scos
    # data_record([np.arange(0.15, 1, 0.05)], s1, 'out/exp_3437_attr.txt')
    # data_record([np.arange(0.15, 1, 0.05)], s3, 'out/exp_3437_score.txt')


def line_line():
    sec = ['aensl-538']
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for i in np.arange(0.35, 1, 0.05):
        print i
        stat, ress, reff, scos, fss = experiment('1684', sec, 1, 1, i)
        s1.append(stat)
        s2.append(ress)
        s3.append(scos)
        s4.append(fss)
    # data_record([np.arange(0.35, 1, 0.05)], s1, 'out/exp_1684_attr_commons.txt')
    # data_record([np.arange(0.05, 1, 0.05)], s2, 'performance3.txt')
    # data_record([np.arange(0.35, 1, 0.05)], s3, 'out/exp_1684_score_commons.txt')
    data_record([np.arange(0.35, 1, 0.05)], s4, 'out/exp_1684_over.txt')

def experiment_attack(data, secret_cate):
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
        epsilon[node] = [0 for a in secret]
        ep2[node] = [len(fb.ran.soc_attr_net.neighbors(a)) / float(fb.ran.soc_net.number_of_nodes())
                     for a in secret]
    price = fb.ran.value_of_attribute('equal')

    mode = 'single'

    greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'greedy', mode)
    ddp, stat3 = fb.ran.d_knapsack_mask(secrets, price, epsilon, 'dp', mode)
    s_good, stat5 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp', mode)
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy', mode)


    r = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []

    for ar in np.arange(0, 1, 0.05):
        re = []
        re2 = []
        re3 = []
        re4 = []
        re5 = []
        for n in range(0, 10, 1):
            att_ran = fb.ran.random_sampling(ar)
            _, res, fs = fb.ran.inference_attack(secrets, att_ran, ep2)
            _, res2, fs2 = greedy.inference_attack(secrets, att_ran, ep2)
            _, res3, fs3 = ddp.inference_attack(secrets, att_ran, ep2)
            _, res5, fs5 = s_good.inference_attack(secrets, att_ran, ep2)
            _, res4, fs4 = s_greedy.inference_attack(secrets, att_ran, ep2)
            re.append(res)
            re2.append(res2)
            re3.append(res3)
            re4.append(res4)
            re5.append(res5)
        r.append(np.average(re))
        r2.append(np.average(re2))
        r3.append(np.average(re3))
        r4.append(np.average(re4))
        r5.append(np.average(re5))
    print r, r2, r3, r4, r5
    data_record([np.arange(0, 1, 0.05)], [r, r2, r3, r4, r5], 'out/exp_1684_attack.txt')

if __name__ == '__main__':
    # line_line()
    # shell_bar()
    experiment_attack('1684', ['aensl-538'])

