from 分组质心function import runge_kutta
import numpy as np
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
colors.CSS4_COLORS



if __name__ == '__main__':
    N = 1000
    t = 10000
    v = 0.03
    #w=np.ones((N,),dtype=int)
    #w = np.random.uniform(1, 3, N)
    w = np.ones(N)*-1
    a = np.random.randn(1,3,500)
    b = np.random.uniform(-3,-1,500)
    w = np.concatenate((a,b))
    print(w)
    theta = np.random.uniform(-np.pi, np.pi, N)
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    R, x, y, theta ,L1,L2,center1,center2= runge_kutta(w,theta,t,x,y)

    # center1_x,center1_y=center1
    # center2_x,center2_y=center2
    # center1_x=np.mod(center1_x,10)
    # center1_y=np.mod(center1_y,10)
    # center2_x=np.mod(center2_x,10)
    # center2_y=np.mod(center2_y,10)
    # print('center1:',center1_x,center1_y)
    # print('center2:',center2_x,center2_y)



    theta11 = []
    theta12 = []
    x1 =[]
    x2 =[]
    y1 = []
    y2 = []
    print(L1[t-1])
    print(L2[t-1])
    for i in L1[t-1]:
        theta11.append(theta[i])
        x1.append(x[i])
        y1.append(y[i])
    for i in L2[t-1]:
        theta12.append(theta[i])
        x2.append(x[i])
        y2.append(y[i])
    T = [i for i in range(t)]
    theta1 = np.cos(theta11)
    theta2 = np.sin(theta11)
    theta3 = np.cos(theta12)
    theta4 = np.sin(theta12)
    plt.quiver(x1, y1, theta1, theta2, color='tomato')
    plt.quiver(x2, y2, theta3, theta4, color='darkturquoise')
    # plt.scatter(center1_x,center1_y,s=20,color = 'tomato',linewidths=0.5,edgecolors='black')
    # plt.scatter(center2_x,center2_y,s=20,color = 'darkturquoise',linewidths=0.5,edgecolors='black')
    plt.show()


