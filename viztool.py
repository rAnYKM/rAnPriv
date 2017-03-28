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
# Date: Feb. 25, LABEL_FONT_SIZE17

from __future__ import division
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rc
from ranfig import load_ranfig
plt.style.use('ggplot')
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# rc('text.latex', unicode=True)

plt.rcParams['figure.figsize'] = 14, 10
# print(plt.style.available)

FOUR_ALGORITHMS = ['V-KP', 'InfoGain', 'NaiveBayes', 'Random']
FIVE_ALGORITHMS = ['V-KP-U', 'InfoGain', 'NaiveBayes', 'Random']
FOUR_NAMES = ['EPPD', 'd-KP', 'NB', 'Random']
FIVE_NAMES = ['EPPD', 'd-KP', 'NB', 'Random']
MARKERS = ['o', 'D', 's', '^']
LINES = ['-', '--', '-.', ':']
MARKER_DICT = {name: MARKERS[index] for index, name in enumerate(FOUR_NAMES)}
LINE_DICT = {name: LINES[index] for index, name in enumerate(FOUR_NAMES)}
MLS_ORG = ['dt', 'rf', 'lr', 'nb']
MLS = [item + '-o' for item in MLS_ORG] + MLS_ORG
MLS_NAMES = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Naive Bayes']
MLS_NAMES_ALL = [item + ' (Prev.)' for item in MLS_NAMES] + [item + ' (Post.)' for item in MLS_NAMES]
RAN_DEFAULT_OUTPUT = '/Users/jiayichen/ranproject/'
CLF_ORG = ['wvrn', 'cdrn-norm-cos', 'nolb-lr-count']
CLF = [item + '-o' for item in CLF_ORG] + CLF_ORG
CLF_NAMES = ['WVRN', 'CDRN', 'NOLB']
METRICS = ['Precision', 'Recall', 'F-Score']
LABEL_FONT_SIZE = 36
NUM_FONT_SIZE = 32
P_EQUAL = r'Disclosure Rate ($p=1$)'
P_UNIQUE = r'Uniqueness Score ($p=p_U$)'
P_COMMONNESS = r'Commonness Score ($p=p_C$)'
P_JACCARD = r'Jaccard Score ($p=p_J$)'
P_AA = r'Adamic/Adar Score ($p=p_A$)'
Y_NAMES_ATTR = [P_EQUAL, P_UNIQUE, P_COMMONNESS]


def attr_cmp_utility(filename, x_name, y_name, seq=FOUR_ALGORITHMS, cat=FOUR_NAMES, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    print(table)
    table = table[seq]
    table.columns = cat
    print(table)
    plt.figure()
    for cate in cat:
        table[cate].plot(linewidth=3, fontsize=NUM_FONT_SIZE, linestyle=LINE_DICT[cate],
                        marker=MARKER_DICT[cate], markersize=16, markeredgewidth=0.0)
    plt.xlabel(x_name, fontsize=LABEL_FONT_SIZE, color='black')
    plt.ylabel(y_name, fontsize=LABEL_FONT_SIZE, color='black')
    plt.legend(fontsize=NUM_FONT_SIZE, bbox_to_anchor=(0., 1.02, 1., .102),
           ncol=4, mode="expand", borderaxespad=0.)
    # ax.set_ylim(0.54, 0.69)
    # ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.grid(b=True, which='major', color='w', linewidth=1)
    # ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    plt.tick_params(colors='black')
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d") )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # plt.show()
    plt.savefig(os.path.join(out_dir, '%s-utility.eps' % exp_no), format='eps', dpi=1000)


def attr_cmp_f1(filename, x_name, y_name, attr, seq=FOUR_ALGORITHMS, cat=FOUR_NAMES, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    table = table[seq + ['Origin']]
    table.columns = cat + ['Origin']
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=NUM_FONT_SIZE)
    ax.set_xlabel(x_name, fontsize=LABEL_FONT_SIZE, color='black')
    ax.set_ylabel(y_name, fontsize=LABEL_FONT_SIZE, color='black')
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def ml_cmp(filename, x_name, y_name, attr, seq=MLS, cat=MLS_NAMES_ALL, exp_no=''):
    table = pd.read_csv(filename, index_col=0)
    table = table[seq]
    table.columns = cat
    plt.figure()
    ax = table.plot(linewidth=2, fontsize=NUM_FONT_SIZE)
    ax.set_xlabel(x_name, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_name, fontsize=LABEL_FONT_SIZE)

    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def ml_cmp_b(filename, x_name, y_name, attr, seq=MLS_ORG, cat=MLS_NAMES, exp_no=''):
    def to_percent(x, pos=0):
        return 100 * x
    table = pd.read_csv(filename, index_col=0)
    table = attack_decrease_rate_table(table, org=seq)
    table = table[seq]
    table.columns = cat
    plt.figure()
    ax = table.plot(linewidth=2.5, fontsize=NUM_FONT_SIZE)
    # ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_xlabel(x_name, fontsize=LABEL_FONT_SIZE, color='black')
    ax.set_ylabel(y_name, fontsize=LABEL_FONT_SIZE, color='black')
    ax.legend(fontsize=NUM_FONT_SIZE, loc='lower left') # , bbox_to_anchor=(0., 1.02, 1., .102),
    #           ncol=4, mode="expand", borderaxespad=0.)
    # ax.set_ylim(0.54, 0.69)
    ax.set_ylim([0.0, 1.01])
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='w', linewidth=1)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax.tick_params(colors='black')

    # ax.margins(y=0.05)
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s-%s.eps' % (exp_no, attr)), format='eps', dpi=1000)


def expr_attr_equal(data_dir):
    filename = 'utility.csv'
    x_name = '$\delta$'
    y_name = P_EQUAL
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    attrs = ['aenslid-538', 'aby-5', 'ahnid-84', 'aencnid-14'] #, 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, exp_no=data_dir + '-e')
    """
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
        attr_cmp_f1(file, x_name, 'F-Score', attr, exp_no=data_dir)
    """


def expr_attr_unique(data_dir):
    filename = 'utility-C.csv'
    x_name = '$\delta$'
    y_name = P_COMMONNESS
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    attrs = ['aenslid-538', 'aby-5', 'ahnid-84', 'aencnid-14'] #, 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir + '-c')
    """
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
        attr_cmp_f1(file, x_name, 'F-Score', attr, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir)
    """

def expr_edge_equal(data_dir):
    filename = 'utility.csv'
    x_name = '$\delta$'
    y_name = P_EQUAL
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, filename)
    # attrs = ['aensl-50', 'aby-5', 'ahnid-84', 'aencnid-14']  # , 'alnid-617'
    attr_cmp_utility(file, x_name, y_name, FOUR_ALGORITHMS, FOUR_NAMES, exp_no=data_dir + '-e')
    # for attr in attrs:
    #     file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s-(f1).csv' % attr)
    #     attr_cmp_f1(file, x_name, 'F-Score', attr, FIVE_ALGORITHMS, FIVE_NAMES, exp_no=data_dir)


def expr_attack(data_dir):
    x_name = '$\delta$'
    y_name = 'Decrease Rate of F-Score'
    attrs = ['aenslid-538']#, 'aby-5', 'ahnid-84', 'aencnid-14']  # , 'alnid-617'
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
    attrs = ['aenslid-538']  # , 'alnid-617'
    for attr in attrs:
        file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s.csv' % attr)
        ml_cmp_b(file, x_name, y_name, attr, seq=CLF_ORG, cat=CLF_NAMES, exp_no=data_dir)

def orignal_attack_effect(data_dir):
    def to_percent(x, pos=0):
        return 100 * x
    y_name = 'Percentage'
    attr = 'aenslid-538'
    file = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, '%s.csv' % attr)
    table = pd.read_csv(file, index_col=0)
    table = table[METRICS]
    plt.figure()

    ax = table.plot(kind='bar', fontsize=NUM_FONT_SIZE)
    plt.xticks(rotation=10)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax.set_xlabel('Local Classifiers', fontsize=LABEL_FONT_SIZE, color='black')
    ax.set_ylabel('Percentage (\%)', fontsize=LABEL_FONT_SIZE, color='black')
    ax.legend(fontsize=NUM_FONT_SIZE, bbox_to_anchor=(0., 1.02, 1., .102),
              ncol=4, mode="expand", borderaxespad=0.)
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
    ax.tick_params(colors='black')
    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, '%s.eps' % (attr)), format='eps', dpi=1000)


def attr_group_plot(data_dir):
    x_name = '$\delta'
    filenames = ['utility.csv', 'utility-U.csv', 'utility-C.csv']
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ax = axes
    for index, file in enumerate(filenames):
        filename = os.path.join(RAN_DEFAULT_OUTPUT, data_dir, file)
        table = pd.read_csv(filename, index_col=0)
        table = table[FIVE_ALGORITHMS]
        table.columns = FIVE_NAMES
        print(table)
        table.plot(ax=ax[index], linewidth=2, fontsize=NUM_FONT_SIZE)
        ax[index].set_xlabel(x_name, fontsize=LABEL_FONT_SIZE, color='black')
        ax[index].set_ylabel(Y_NAMES_ATTR[index], fontsize=LABEL_FONT_SIZE, color='black')
        ax[index].tick_params(colors='black')
        ax[index].legend()
    plt.legend(fontsize=NUM_FONT_SIZE, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=4, mode="expand", borderaxespad=0.)

    out_dir = os.path.join(load_ranfig()['OUT'], datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.show()

if __name__ == '__main__':
    # expr_attr_equal('gp-attr')
    # expr_attr_unique('gp-attr')
    # expr_attack('fb-attr')
    # expr_edge_equal('res3NUM_FONT_SIZE')
    # expr_attack_relation('res319')
    expr_edge_equal('res324')
    # attr_group_plot('fb-attr')
    # orignal_attack_effect('res317-2')
