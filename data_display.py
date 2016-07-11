# This script is used to visualize all data

import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt

def July10_01():
    with open('edge_reduce.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x,y

        lines = plt.plot(x, y, linewidth=1.5)
        plt.axis([0.05, 0.95, 0.3, 0.75], fontsize=14)
        plt.xlabel(r'Security Threshold', fontsize=16)
        plt.ylabel(r'Percentage of Masked Attributes', fontsize=16)
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0.3, 0.8, 0.05))
        plt.legend(lines, ('Random Mask', 'd-KP', 'DP', 'Primal-greedy', 'Dual-greedy'), loc='lower left')
        plt.grid(True)
        # plt.show()
        plt.savefig("test.eps", format="eps")

def July10_02():
    with open('performance.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x, y

        lines = plt.plot(x, y, linewidth=1.5)
        plt.axis([0.05, 0.95, 0.5, 0.52], fontsize=14)
        plt.xlabel(r'Security Threshold', fontsize=16)
        plt.ylabel(r'Percentage of Masked Attributes', fontsize=16)
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0.2, 0.55, 0.05))
        plt.legend(lines, ('Random Mask', 'd-KP', 'DP', 'Primal-greedy', 'Dual-greedy'), loc='upper left')
        plt.grid(True)
        plt.show()
        # plt.savefig("test.eps", format="eps")


if __name__ == '__main__':
    July10_02()