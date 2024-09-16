'''
from VKfunction import runge_kutta
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

N=100
w = np.random.uniform(-1, 1, N)
a=np.ones((N,),dtype=int)
print(type(w))
print(type(a)）
'''

#存档
from 分组质心function import runge_kutta
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import colors
colors.CSS4_COLORS


if __name__ == '__main__':
    N = 1000
    t = 50000
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
    R, x, y, theta = runge_kutta(w,theta,t,x,y)
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

    '''import numpy as np
    import numba as nb
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors

    colors.CSS4_COLORS

    x = np.arange(0, 12.1, 0.1)
    y = np.sin(x)
    plt.plot(x, y, color='cornflowerblue', )
    plt.show()'''