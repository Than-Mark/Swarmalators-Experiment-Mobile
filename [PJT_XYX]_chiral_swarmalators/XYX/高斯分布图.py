# -*-coding:utf-8-*-
"""
python绘制标准正态分布曲线
"""
# ==============================================================
import numpy as np
import math
import matplotlib.pyplot as plt


def gd(x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值
    Argument:
      x: array
        输入数据（自变量）
      mu: float
        均值
      sigma: float
        方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu) ** 2 / (2 * sigma))
    return left * right


if __name__ == '__main__':
    # 自变量
    x = np.arange(-8, 8, 0.1)
    # 因变量（不同均值或方差）
    y_1 = gd(x, 2, 4)
    y_2 = gd(x, -2, 4)
    y_3 = gd(x, 2, 2)
    y_4 = gd(x, -2, 2)
    y_5 = gd(x, 2, 1)
    y_6 = gd(x, -2, 1)
    y_7 = gd(x, 2, 0.5)
    y_8 = gd(x, -2, 0.5)
    y_9 = gd(x, 2, 0.2)
    y_10 = gd(x, -2, 0.2)
    # 绘图
    plt.plot(x, y_1, color='green')
    plt.plot(x, y_2, color='green')
    plt.plot(x, y_3, color='blue')
    plt.plot(x, y_4, color='blue')
    plt.plot(x, y_5, color='red')
    plt.plot(x, y_6, color='red')
    plt.plot(x, y_7, color='orange')
    plt.plot(x, y_8, color='orange')
    plt.plot(x, y_9, color='brown')
    plt.plot(x, y_10, color='brown')
    # 设置坐标系
    plt.xlim(-8.0, 8.0)
    plt.ylim(-0.2, 1.0)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    #plt.legend(labels=['$\mu = -3, \sigma^2=0.5$', '$\mu = 3, \sigma^2=0.5$'])
    plt.legend(labels=['$\mu = 2, \sigma^2=4$','$\mu = -2, \sigma^2=4$', '$\mu = 2, \sigma^2=2$','$\mu = -2, \sigma^2=2$','$\mu = 2, \sigma^2=1$','$\mu = -2, \sigma^2=1$','$\mu = 2, \sigma^2=0.5$','$\mu = -2, \sigma^2=0.5$','$\mu = 2, \sigma^2=0.2$','$\mu = -2, \sigma^2=0.2$'])
    #plt.legend(labels=['$\mu = 0, \sigma^2=0.2$', '$\mu = 0, \sigma^2=1.0$', '$\mu = 0, \sigma^2=5.0$'])
    plt.show()