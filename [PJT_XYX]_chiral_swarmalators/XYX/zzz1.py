from movie_function import runge_kutta
import numpy as np
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as ma

colors.CSS4_COLORS


if __name__ == '__main__':
    N = 1000
    t = 5000
    v = 0.03
    #w=np.ones((N,),dtype=int)
    #w = np.random.uniform(1, 3, N)
    a = np.random.uniform(1,3,500)
    b = np.random.uniform(-3,-1,500)
    w = np.concatenate((a,b))
    print(w)
    theta = np.random.uniform(-np.pi, np.pi, N)
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    R, x, y, theta,X,Y,Theta = runge_kutta(w,theta,t,x,y)
    T = [i for i in range(t)]
    theta1 = np.cos(Theta)
    theta2 = np.sin(Theta)

    fig, ax = plt.subplots()

    LN= []
    for i in range(t):

        plt.quiver(X[i],Y[i],theta1[i],theta2[i])
        # ln = plt.scatter(X[i],Y[i])
        plt.pause(0.1)
        plt.cla()

    plt.show()