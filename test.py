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
    price = fb.ran.value_of_attribute('common')
    att_ran = fb.ran.random_sampling(ar)
    mode = 'double'
    score_li = []
    stat_li = []
    res_li = []
    fs_li = []
    for times in range(0, 10, 1):
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
    # s_good, stat5 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp', mode)
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy', mode)
    compare, stat6, _ = fb.ran.nb_masking(secrets, ep2)
    _, res, fs = fb.ran.inference_attack(secrets, att_ran, ep2)
    _, res2, fs2 = greedy.inference_attack(secrets, att_ran, ep2)
    _, res3, fs3 = ddp.inference_attack(secrets, att_ran, ep2)
    # _, res5, fs5 = s_good.inference_attack(secrets, att_ran, ep2)
    _, res4, fs4 = s_greedy.inference_attack(secrets, att_ran, ep2)
    _, res6, fs6 = compare.inference_attack(secrets, att_ran, ep2)

    _, max_score = fb.ran.utility_measure(secrets, price, mode)
    _, score2 = greedy.utility_measure(secrets, price, mode)
    _, score3 = ddp.utility_measure(secrets, price, mode)
    # _, score5 = s_good.utility_measure(secrets, price, mode)
    _, score4 = s_greedy.utility_measure(secrets, price, mode)
    _, score6 = compare.utility_measure(secrets, price, mode)
    stat = [stat1, stat2, stat3, stat4, stat6]
    ress = [res1, res2, res3, res4, res6]
    scos = [score1, score2, score3, score4, score6]
    n_scos = [i / float(max_score) for i in scos]
    reff = res
    print n_scos
    print fs, fs1, fs2, fs3, fs4, fs6
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
    # price2 = fb.ran.value_of_relation('equal')
    price3 = fb.ran.value_of_edge('AA')
    # print price3
    att_ran = fb.ran.random_sampling(ar)
    def_ran, _, stat1 = fb.ran.adv_random_masking(secrets, ep2, 'on')
    # greedy, stat2 = fb.ran.d_knapsack_mask(secrets, price, epsilon)
    # greedy2, stat3 = fb.ran.d_knapsack_relation(secrets, price2, epsilon)
    ggd, stat2 = fb.ran.d_knapsack_relation_global(secrets, price3, epsilon)
    s_greedy, stat3 = fb.ran.s_knapsack_relation_global(secrets, price3, ep2)
    nb, _, stat4 = fb.ran.nb_masking(secrets, ep2, 'on')
    _, res1, fs1 = def_ran.inference_attack_relation(secrets, att_ran, ep2)
    _, res, fs = fb.ran.inference_attack_relation(secrets, att_ran, ep2)
    _, res2, fs2 = ggd.inference_attack_relation(secrets, att_ran, ep2)
    # _, res3 = greedy2.inference_attack_relation(secrets, att_ran)
    _, res3, fs3 = s_greedy.inference_attack_relation(secrets, att_ran, ep2)
    _, res4, fs4 = nb.inference_attack_relation(secrets, att_ran, ep2)

    _, score = fb.ran.relation_utility_measure(price3)
    _, score1 = def_ran.relation_utility_measure(price3)
    _, score2 = ggd.relation_utility_measure(price3)
    _, score3 = s_greedy.relation_utility_measure(price3)
    _, score4 = nb.relation_utility_measure(price3)

    print res, res1, res2, res3, res4
    print fs, fs1, fs2, fs3, fs4
    stat = [stat1, stat2, stat3, stat4]
    print stat
    ress = [res1, res2, res3, res4]
    fss = [fs1, fs2, fs3, fs4]
    reff = res
    scos = [score1, score2, score3, score4]
    n_scos = [i / float(score) for i in scos]
    return stat, ress, reff, n_scos, fss


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


def single_experiment_strict_point(data, secret_cate, ar, dr, e):
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
    compare, stat6, _ = fb.ran.nb_masking(secrets, ep2)
    _, res6, fs6 = compare.inference_attack(secrets, att_ran, ep2)
    _, score6 = compare.utility_measure(secrets, price, 'single')
    _, max_score = fb.ran.utility_measure(secrets, price, 'single')
    return 1 - score6/float(max_score)


def shell_bar():
    sec = ['aensl-537']
    s1 = []
    s2 = []
    s3 = []
    # stat, ress, reff, scos = experiment_strict_point('1684', sec, 1, 1, 0.35)
    scos = single_experiment_strict_point('3437', sec, 1, 1, 0.35)
    # s1.append(stat)
    # s2.append(ress)
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
    data_record([np.arange(0.35, 1, 0.05)], s1, 'data/exp_1684c_attr_new.txt')
    # data_record([np.arange(0.05, 1, 0.05)], s2, 'performance3.txt')
    data_record([np.arange(0.35, 1, 0.05)], s3, 'data/exp_1684c_score_new.txt')
    data_record([np.arange(0.35, 1, 0.05)], s4, 'data/exp_1684c_over_new.txt')


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
    # s_good, stat5 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'dp', mode)
    s_greedy, stat4 = fb.ran.s_knapsack_mask(secrets, price, ep2, 'greedy', mode)
    compare, stat6, _ = fb.ran.nb_masking(secrets, ep2)

    r = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    r6 = []

    for ar in np.arange(0, 1, 0.05):
        re = []
        re2 = []
        re3 = []
        re4 = []
        re5 = []
        re6 = []
        for n in range(0, 10, 1):
            att_ran = fb.ran.random_sampling(ar)
            _, res, fs = fb.ran.inference_attack(secrets, att_ran, ep2)
            _, res2, fs2 = greedy.inference_attack(secrets, att_ran, ep2)
            _, res3, fs3 = ddp.inference_attack(secrets, att_ran, ep2)
            # _, res5, fs5 = s_good.inference_attack(secrets, att_ran, ep2)
            _, res4, fs4 = s_greedy.inference_attack(secrets, att_ran, ep2)
            _, res6, fs6 = compare.inference_attack(secrets, att_ran, ep2)
            re.append(res)
            re2.append(res2)
            re3.append(res3)
            re4.append(res4)
            # re5.append(res5)
            re6.append(res6)
        r.append(np.average(re))
        r2.append(np.average(re2))
        r3.append(np.average(re3))
        r4.append(np.average(re4))
        # r5.append(np.average(re5))
        r6.append(np.average(re6))
    print r, r2, r3, r4, r6
    data_record([np.arange(0, 1, 0.05)], [r, r2, r3, r4, r6], 'out/exp_1684_attack_six.txt')


def line_relation():
    sec = ['aensl-537']
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for i in np.arange(0.2, 1, 0.1):
        print i
        stat, ress, reff, scos, fss = experiment_relation('3437', sec, 1, 1, i)
        s1.append(stat)
        s2.append(ress)
        s3.append(scos)
        s4.append(fss)
    data_record([np.arange(0.2, 1, 0.1)], s1, 'out/exp_3437e_attr_4a.txt')
    # data_record([np.arange(0.05, 1, 0.05)], s2, 'performance3.txt')
    data_record([np.arange(0.2, 1, 0.1)], s3, 'out/exp_3437e_score_4a.txt')
    data_record([np.arange(0.2, 1, 0.1)], s4, 'out/exp_3437e_over_4a.txt')


if __name__ == '__main__':
    # line_line()
    # shell_bar()
    # experiment_attack('1684', ['aensl-538'])
    # experiment_relation('0', ['aensl-50'], 1, 1, 0.5)
    line_relation()


