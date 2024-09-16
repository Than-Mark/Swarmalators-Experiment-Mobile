from Kfunction_movie import runge_kutta
import numpy as np
import numba as nb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as ma

colors.CSS4_COLORS

plt.rcParams['animation.ffmpeg_path'] = r'D:\study software\python project\ffmpeg-2023-05-22-git-877ccaf776-essentials_build(1)\ffmpeg-2023-05-22-git-877ccaf776-essentials_build\bin\ffmpeg.exe'
if __name__ == '__main__':
    N = 1000
    t = 50000
    v = 0.03
    #w=np.ones((N,),dtype=int)
    #w = np.random.uniform(1, 3, N)
    a = np.random.uniform(1,3,950)
    b = np.random.uniform(-3,-1,50)
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
    line = plt.quiver(X[0],Y[0],theta1[0],theta2[0])

    def update(frame):
        plt.cla()
        line = plt.quiver(X[frame],Y[frame],theta1[frame],theta2[frame])
        return line


    ani = ma.FuncAnimation(fig, update, frames=np.arange(0,t,1), interval=10, repeat=False)

    ani.save('Pendulum_Animation.mp4')
    plt.show()