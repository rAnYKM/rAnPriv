# This script is used to visualize all data

import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def July10_01():
    with open('out/exp_1684_score_unique.txt', 'r') as fp:
        x = [float(i) for i in fp.readline().strip().split()]
        fp.readline()
        li = fp.readline()
        y = []
        while li:
            y.append([float(i) for i in li.strip().split()])
            li = fp.readline()
        print x,y

        cm = plt.get_cmap('viridis')
        cNorm = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        print scalarMap.get_clim()
        # plt.style.use("ggplot")
        lines = plt.plot(x, y, linewidth=2)
        plt.setp(lines[0], marker='o', color=scalarMap.to_rgba(1), markersize=8, linewidth=2.0)
        plt.setp(lines[1], marker='v', color=scalarMap.to_rgba(2), markersize=8, linewidth=2.0)
        plt.setp(lines[2], marker='^', color=scalarMap.to_rgba(3), markersize=8, linewidth=2.0)
        plt.setp(lines[3], marker='s', color=scalarMap.to_rgba(4), markersize=8, linewidth=2.0)
        plt.setp(lines[4], marker='d', color=scalarMap.to_rgba(5), markersize=8, linewidth=2.0)
        # plt.axis([0.35, 0.95, 0.08, 0.45])
        plt.xlabel(r'Protection Threshold', fontsize=20)
        plt.ylabel(r'Percentage of Masked Attributes', fontsize=20)
        # plt.xticks(np.arange(0.35, 1.0, 0.1), fontsize=20)
        # plt.yticks(np.arange(0.1, 0.45, 0.05), fontsize=20)
        plt.legend(lines,
                    ('Random Mask', 'd-KP(Greedy)', 'd-KP(DP)', 'Greedy', 'DP'), loc='upper right')
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
        plt.xlabel(r'Security Threshold ', fontsize=16)
        plt.ylabel(r'Percentage of Masked Attributes ', fontsize=16)
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0, 0.55, 0.05))
        plt.legend(lines, ('Random Mask', 'd-KP', 'DP', 'Primal-greedy', 'Dual-greedy'), loc='upper left')
        plt.grid(True)
        plt.show()
        # plt.savefig("test.eps", format="eps")

def July12_01():
    with open('out/exp_3437_score.txt', 'r') as fp:
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

def July22_01_bar():
    width = 0.15
    pos = list(range(8))
    pos = [p - 2.5 * width for p in pos]
    # print pos
    cm = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    y0 = [[0.57380525686977313, 0.5086618876941458, 0.5086618876941458, 0.5038829151732378, 0.5232974910394266],
          [0.33484400101462752, 0.23057411008708886, 0.22178067134522703, 0.23522448634480425, 0.24782277838843325],
          [0.43027522935779816, 0.32360300250208507, 0.32068390325271057, 0.32068390325271057, 0.3573811509591326],
          [0.34693486590038314, 0.22924648786717752, 0.21839080459770116, 0.24521072796934865, 0.26500638569604085],
          [0.32412412412412406, 0.2702702702702703, 0.2702702702702703, 0.26326326326326327, 0.28928928928928926],
          [0.410137810866466, 0.353397750673214, 0.34579439252336447, 0.33914145414224617, 0.3592586725803897],
          [0.373729233820977, 0.2793206050086784, 0.2788246962558889, 0.2788246962558889, 0.3000247954376395],
          [0.2779732582688248, 0.24466338259441708, 0.24466338259441708, 0.2437250762373915, 0.2531081398076472]]
    y = np.matrix([[1 - i for i in j] for j in y0])
    y = y.transpose().tolist()
    x = ['#0', '#107', '#348', '#414', '#686', '#1684', '#1912', '#3437']
    xx = ('Random Mask', 'd-KP(Greedy)', 'd-KP(DP)', 'Greedy', 'DP')
    plt.figure(figsize=(16, 5))
    bars = list()
    for i in range(5):
        bar = plt.bar([p + width*i for p in pos],
        y[i],
        width,
        color=scalarMap.to_rgba(i + 1),
        label=xx[i]
        )
        bars.append(bar)
    plt.xlim(min(pos) - width, max(pos) + width * 6)
    plt.ylim([0.3, 0.85])
    plt.xlabel('Ego Network #', fontsize=22)
    plt.ylabel('Self-Disclosure (p=1)', fontsize=22)
    plt.grid(True)
    plt.xticks(range(8), x, fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc='lower right', fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig("data/longcomparebar.eps", format="eps")

if __name__ == '__main__':
    July10_01()
    # July22_01_bar()