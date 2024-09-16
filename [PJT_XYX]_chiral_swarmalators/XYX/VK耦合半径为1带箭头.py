from VKfunction import runge_kutta
import numpy as np
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
colors.CSS4_COLORS



if __name__ == '__main__':
    N = 1000
    t = 50000
    v = 0.03
    # 1.单频(F1):
    #w=np.ones((N,),dtype=int)

    # 2.两个频率(F2）：
    #a=np.ones(500,)
    #b=np.ones(500,)*(-1)

    # 3.单一均匀分布（F3）：
    #w=np.random.uniform(1, 3, N)

    # 4.双均匀分布（F4）：
    a = np.random.uniform(1,3,667)
    b = np.random.uniform(-3,-1,333)

    # 5.单高斯分布（F4）均值为0：
    #w = np.random.normal(loc=0, scale=1.0, size=N)

    # 6.双高斯分布（F4）：
    #a=np.random.normal(loc=3, scale=0.5, size=500)
    #b=np.random.normal(loc=-3, scale=0.5, size=500)
    w = np.concatenate((a,b))
    print(w)
    theta = np.random.uniform(-np.pi, np.pi, N)
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    R, x, y, theta = runge_kutta(w,theta,t,x,y)
    T = [i for i in range(t)]
    theta1 = np.cos(theta)
    theta2 = np.sin(theta)

    #画图

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 1, 2)
    #颜色
    ax1.quiver(x, y, theta1, theta2,color='darkorange')
    ax2.scatter(x, w,s=0.5,color='limegreen')
    ax3.scatter(x,theta,s=0.5,color='lightseagreen')
    ax4.scatter(T, R,s=0.5,color='royalblue')
    #字体，标签
    ax1.set_xlabel('x',fontname='Times New Roman')
    ax1.set_ylabel('y',fontname='Times New Roman')
    ax2.set_xlabel('x',fontname='Times New Roman')
    ax2.set_ylabel('$\omega$',fontname='Times New Roman')
    ax3.set_xlabel('x',fontname='Times New Roman')
    ax3.set_ylabel('$\Theta$',fontname='Times New Roman')
    ax4.set_xlabel('t',fontname='Times New Roman')
    ax4.set_ylabel('R',fontname='Times New Roman')
    #坐标轴字体大小
    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)
    ax3.tick_params(labelsize=10)
    ax4.tick_params(labelsize=10)
    #坐标轴字体
    labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels1]
    labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels2]
    labels3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels3]
    labels4 = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels4]
    #设置图4的横轴范围
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 10])
    ax2.set_xlim([0, 10])
    ax2.set_ylim([-3, 3])
    ax3.set_xlim([0, 10])
    ax3.set_ylim([-3, 3])
    ax4.set_xlim([0,50000])
    #调整位置，标签不被遮挡
    plt.tight_layout()
    plt.figure()
    plt.quiver(x, y, theta1, theta2, color='darkorange')
    plt.figure()
    plt.scatter(T, R, s=0.5, color='royalblue')
    plt.xlabel('t',fontname='Times New Roman')
    plt.ylabel('R',fontname='Times New Roman')
    plt.show()


