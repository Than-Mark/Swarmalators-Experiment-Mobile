from 分组质心function import runge_kutta_K_exp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 1000
    t = 50000
    v = 0.03
    #w = np.random.uniform(-1, 1, N)
    w = np.random.uniform(1, 3, N)
    #a = np.random.uniform(1, 3, 500)
    #b = np.random.uniform(-3, -1, 500)
    #w = np.concatenate((a, b))
    theta = np.random.uniform(-np.pi, np.pi, N)
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    R, x, y, theta = runge_kutta_K_exp(w,theta,t,x,y)
    T = [i for i in range(t)]
    plt.figure()
    plt.scatter(T, R, s=1)
    plt.figure()
    plt.scatter(x,y)
    plt.figure()
    plt.scatter(x, w)
    plt.figure()
    plt.scatter(x,theta)
    plt.show()
