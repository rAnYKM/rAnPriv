# This script is used to visualize all data

import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt

def auto_scale

def July10_01():
    with open('out/exp_686_attr.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x,y

        lines = plt.plot(x, y, 'o-', linewidth=2, markersize=6)
        plt.axis([0.3, 0.95, 0.1, 0.45], fontsize=20)
        plt.xlabel(r'Security Threshold', fontsize=20)
        plt.ylabel(r'Percentage of Masked Attributes', fontsize=20)
        plt.xticks(np.arange(0.3, 1, 0.05))
        plt.yticks(np.arange(0.1, 0.45, 0.05))
        plt.legend(lines, ('Random Mask', 'd-KP(Greedy)', 'd-KP(DP)', 'DP', 'Greedy'), loc='lower left')
        plt.grid(True, color='black', alpha=0.5)
        plt.show()
        # plt.savefig("test4.eps", format="eps")

def July10_02():
    with open('performance_different_attacker.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x, y

        lines = plt.plot(x, y, linewidth=1.5)
        plt.axis([0.05, 0.95, 0, 0.52], fontsize=14)
        plt.xlabel(r'Security Threshold', fontsize=16)
        plt.ylabel(r'Percentage of Masked Attributes', fontsize=16)
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0, 0.55, 0.05))
        plt.legend(lines, ('Random Mask', 'd-KP', 'DP', 'Primal-greedy', 'Dual-greedy'), loc='upper left')
        plt.grid(True)
        plt.show()
        # plt.savefig("test.eps", format="eps")

def July12_01():
    with open('out/exp_348_score.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x, y

        lines = plt.plot(x, y, linewidth=1.5)
        plt.axis([0.05, 0.95, 0.2, 0.95], fontsize=14)
        plt.xlabel(r'Security Threshold', fontsize=16)
        plt.ylabel(r'Average Score (Commonness)', fontsize=16)
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0.35, 0.95, 0.1))
        plt.legend(lines, ('Random Mask', 'd-KP', 'DP', 'Primal-greedy', 'Dual-greedy'), loc='upper left')
        plt.grid(True)
        plt.show()
        # plt.savefig("test5.eps", format="eps")

if __name__ == '__main__':
    July10_01()