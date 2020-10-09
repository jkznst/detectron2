# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def plot_speed():
    plt.axvspan(25, 75, facecolor='#2ca02c', alpha=0.3)

    plt.scatter(2, 50.2, s=100, label='[82] w/ ref.')
    plt.scatter(3, 62.7, s=100, marker='^', label='BB8[87] w/ ref.')
    plt.scatter(10, 79, s=100, marker='s', label='SSD-6D[84] w/ ref.')
    plt.scatter(50, 56, s=100, marker='*', label='YOLO6D[88]')
    plt.scatter(6, 62.7, s=100, marker='p', c='black', label='PoseCNN[86]')
    plt.scatter(4.5, 64.7, s=100, marker='v', c='yellow', label='AAE[94] w/ ref.')

    # CRPNet
    plt.scatter(30, 74.7, s=100, c='purple', marker='D', label='CRPNet(ours)')
    plt.scatter(35, 73.6, s=100, c='purple', marker='D')
    plt.scatter(43, 70.3, s=100, c='purple', marker='D')
    x = [30, 35, 43]
    y = [74.7, 73.6, 70.3]
    plt.plot(x, y, c='purple')

    plt.xlabel('FPS')
    plt.ylabel('ADD@0.1(%)')
    plt.xlim(0, 75)
    # plt.legend(['[82]', 'BB8[87]', 'SSD-6D[84]', 'YOLO6D[88]', 'CRPNet(ours)'])
    plt.legend()
    plt.show()

def plot_indirect_strategy():
    x = ['ape', 'bvise.', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holep.', 'iron', 'lamp', 'phone']
    direct = [93.5, 90.3, 93.3, 93.7, 94.5, 93.5, 94.4, 93.2, 94.0, 92.2, 88.8, 86.2, 92.5]
    indirect = [96.5, 93.4, 96.4, 97.6, 97.4, 96.6, 97.1, 96.1, 96.9, 94.9, 91.5, 88.1, 95.3]

    plt.bar(x, indirect, label='Cascade Regression Strategy (ours)', color='orange')
    plt.bar(x, direct, label="Direct Regression Strategy [88]", color='blue')

    plt.xticks(np.arange(len(x)), x, rotation=320, fontsize=10)
    plt.legend(loc='upper left', fontsize=10)
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ylabel('REP@5px(%)')
    # plt.xlabel('Objects')
    plt.ylim(85, 100)
    plt.show()

if __name__ == "__main__":
    plot_speed()