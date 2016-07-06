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
    print "hello world"
    fb = FacebookEgoNet(data)
    # Build secret dict
    secrets = dict()
    epsilon = dict()
    for node in fb.ran.soc_net.nodes():
        feature = [attr for attr in fb.ran.soc_attr_net.neighbors(node)
                   if attr[0] == 'a']
        secret = [attr for attr in feature if attr.split('-')[0] in secret_cate]
        secrets[node] = secret
        epsilon[node] = [np.log2(e) -
                         np.log2(len(fb.ran.soc_attr_net.neighbors(a))/float(fb.ran.soc_net.number_of_nodes()))
                         for a in secret]
    att_ran = fb.ran.random_sampling(ar)
    def_ran = fb.ran.random_masking(secrets, dr)
    greedy = fb.ran.d_knapsack_mask(secrets, epsilon)
    greedy2 = greedy.d_knapsack_relation(secrets, epsilon)
    print def_ran.inference_attack(secrets, def_ran)
    print greedy.inference_attack(secrets, greedy)
    _, res = fb.ran.inference_attack(secrets, att_ran)
    _, res2 = def_ran.inference_attack(secrets, att_ran)
    _, res4 = def_ran.inference_attack_relation(secrets, att_ran)
    _, res3 = fb.ran.inference_attack_relation(secrets, att_ran)
    _, res5 = greedy.inference_attack(secrets, att_ran)
    _, res6 = greedy2.inference_attack_relation(secrets, att_ran)
    print res, res2, res3, res4, res5, res6

if __name__ == '__main__':
    sec = ['aensl', 'aencn']
    experiment('0', sec, 0.7, 0.3, 1)
