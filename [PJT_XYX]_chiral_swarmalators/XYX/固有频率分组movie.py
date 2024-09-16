from 固有频率分组movie_function import runge_kutta
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

    # 1.单频(F1):
    #w=np.ones((N,),dtype=int)

    # 2.两个频率(F2）：
    #a=np.ones(500,)
    #b=np.ones(500,)*(-1)

    # 3.单一均匀分布（F3）：
    #w=np.random.uniform(1, 3, N)

    # 4.双均匀分布（F4）：
    #a = np.random.uniform(1,3,800)
    #b = np.random.uniform(-3,-1,200)

    # 5.单高斯分布（F4）均值为0：
    #w = np.random.normal(loc=0, scale=1.0, size=N)

    # 6.双高斯分布（F4）：
    a=np.random.normal(loc=2, scale=4, size=500)
    b=np.random.normal(loc=-2, scale=4, size=500)
    w = np.concatenate((a,b))
    print(w)
    theta = np.random.uniform(-np.pi, np.pi, N)
    x = np.random.uniform(0, 10, N)
    y = np.random.uniform(0, 10, N)
    R, x, y, theta,X1,X2,Y1,Y2,Theta,Theta1_c,Theta1_s,Theta2_c,Theta2_s = runge_kutta(w,theta,t,x,y)
    print(Theta[0])
    print(type(Theta[0]))
    T = [i for i in range(t)]

    fig, ax = plt.subplots()
    line = plt.quiver(X1[0],Y1[0],Theta1_c[0],Theta1_s[0])
    line = plt.quiver(X2[0],Y2[0],Theta2_c[0],Theta2_s[0])

    def update(frame):
        plt.cla()
        line = plt.quiver(X1[frame],Y1[frame],Theta1_c[frame],Theta1_s[frame], color='tomato')
        line = plt.quiver(X2[frame],Y2[frame],Theta2_c[frame],Theta2_s[frame], color='darkturquoise')
        return line


    ani = ma.FuncAnimation(fig, update, frames=np.arange(0,t,1), interval=10, repeat=False)

    ani.save('Pendulum_Animation.mp4')
    plt.show()