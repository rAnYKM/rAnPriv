# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _
#    _____/   |  ____  / __ \_____(_)   __
#   / ___/ /| | / __ \/ /_/ / ___/ / | / /
#  / /  / ___ |/ / / / ____/ /  / /| |/ /
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/
#
# Script Name: viztool.py
# Date: Feb. 25, 2017

from __future__ import division
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from ranfig import load_ranfig
plt.style.use('ggplot')
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# print(plt.style.available)

FOUR_ALGORITHMS = ['V-KP', 'InfoGain', 'NaiveBayes', 'Random']
FIVE_ALGORITHMS = ['V-KP-U', 'InfoGain', 'NaiveBayes', 'Random']
FOUR_NAMES = ['EPPD', 'IG', 'NB', 'Random']
FIVE_NAMES = ['EPPD', 'IG', 'NB', 'Random']
MLS_ORG = ['dt', 'lr', 'nb']
MLS = [item + '-o' for item in MLS_ORG] + MLS_ORG
MLS_NAMES = ['Decision Tree', 'Logistic Regression', 'Naive Bayes']
MLS_NAMES_ALL = [item + ' (Prev.)' for item in MLS_NAMES] + [item + ' (Post.)' for item in MLS_NAMES]
RAN_DEFAULT_OUTPUT = '/Users/jiayichen/ranproject/'
CLF_ORG = ['wvrn', 'nolb-lr-distrib', 'logistic']
CLF = [item + '-o' for item in CLF_ORG] + CLF_ORG
CLF_NAMES = ['WVRN', 'NOLB-LR', 'Logistic']


def attr_cmp_utility(filename, x_name, y_name, seq=FOUR_ALGORITHMS, cat=FOUR_NAMES, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    print(table)
    table = table[seq]
    table.columns = cat
    print(table)
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=16)
    ax.set_xlabel(x_name, fontsize=20)
    ax.set_ylabel(y_name, fontsize=20)
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d") )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-utility.eps' % exp_no), format='eps', dpi=1000)


def attr_cmp_f1(filename, x_name, y_name, attr, seq=FOUR_ALGORITHMS, cat=FOUR_NAMES, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    table = table[seq + ['Origin']]
    table.columns = cat + ['Origin']
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=16)
    ax.set_xlabel(x_name, fontsize=20)
    ax.set_ylabel(y_name, fontsize=20)
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def ml_cmp(filename, x_name, y_name, attr, seq=MLS, cat=MLS_NAMES_ALL, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    table = table[seq]
    table.columns = cat
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=16)
    ax.set_xlabel(x_name, fontsize=20)
    ax.set_ylabel(y_name, fontsize=20)
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def ml_cmp_b(filename, x_name, y_name, attr, seq=MLS_ORG, cat=MLS_NAMES, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    table = attack_decrease_rate_table(table, org=seq)
    table = table[seq]
    table.columns = cat
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=16)
    ax.set_xlabel(x_name, fontsize=20)
    ax.set_ylabel(y_name, fontsize=20)
    ax.set_ylim([0.0, 1.01])
    # ax.margins(y=0.05)
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def expr_attr_equal(data_dir):
    filename = 'utility.csv'
    x_name = '$\delta$'
    y_name = 'Utility ($p=1$)'
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    attrs = ['aenslid-538', 'aby-5', 'ahnid-84', 'aencnid-14'] #, 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, exp_no=data_dir)
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
        attr_cmp_f1(file, x_name, 'F-Score', attr, exp_no=data_dir)


def expr_attr_unique(data_dir):
    filename = 'utility.csv'
    x_name = '$\delta$'
    y_name = 'Utility ($p=p_U$)'
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    attrs = ['aenslid-538', 'aby-5', 'ahnid-84', 'aencnid-14'] #, 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir)
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
        attr_cmp_f1(file, x_name, 'F-Score', attr, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir)


def expr_edge_equal(data_dir):
    filename = 'utility-j.csv'
    x_name = '$\delta$'
    y_name = 'Utility ($p=p_J$)'
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    # attrs = ['aensl-50', 'aby-5', 'ahnid-84', 'aencnid-14']  # , 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, FOUR_ALGORITHMS, FOUR_NAMES, exp_no=data_dir)
    # for attr in attrs:
    #     file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
    #     attr_cmp_f1(file, x_name, 'F-Score', attr, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir)


def expr_attack(data_dir):
    x_name = '$\delta$'
    y_name = 'Decrease Rate of F-Score'
    attrs = ['aenslid-538', 'aby-5', 'ahnid-84', 'aencnid-14']  # , 'alnid-617'
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s.csv' % attr)
        ml_cmp_b(file, x_name, y_name, attr, exp_no=data_dir)


def attack_decrease_rate_table(table, org=MLS_ORG):
    def cal_rate(row, lag):
        return (row[lag + '-o'] - row[lag])/row[lag + '-o']

    # panel = pd.Panel({'prev': table[MLS[:3]], 'post': table[MLS[3:]]})
    new_table = pd.DataFrame(columns=org, index=table.index.values)
    for ml in org:
        new_table[ml] = table.apply(cal_rate, axis=1, lag=ml)
    return new_table

def expr_attack_relation(data_dir):
    x_name = '$\delta$'
    y_name = 'Decrease Rate of F-Score'
    attrs = ['aensl-50']  # , 'alnid-617'
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s.csv' % attr)
        ml_cmp_b(file, x_name, y_name, attr, seq=CLF_ORG, cat=CLF_NAMES, exp_no=data_dir)


if __name__ == '__main__':
    # expr_attr_equal('res225')
    # expr_attr_unique('res226')
    # expr_attack('res226-3')
    # expr_edge_equal('res306')
    expr_attack_relation('res306-3')
