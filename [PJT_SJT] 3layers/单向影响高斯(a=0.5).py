import pylab as plt
import numpy as np
import numba as nb


def ini_condition():
    # 区别于Kuramoto-2000，此时具有相同的自然频率，记成w=
    # theta1 = np.random.uniform(0,2*np.pi,N)
    # theta2 = np.random.uniform(0,2*np.pi,N)

    theta1 = np.ones(N)
    theta2 = np.ones(N)
    K3=np.ones(N)/N
    return np.hstack((theta1)),np.hstack((theta2)),K3

@nb.jit(nopython=True)
def master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1):
    d_theta1 = fre1 +  a*k/N * (np.cos(theta1) * d2 - np.sin(theta1) * d1)+((1-a)*k/(N**2))*((s2+s3)*np.cos(2*theta1)-(s1-s4)*np.sin(2*theta1))
    return d_theta1

@nb.jit(nopython=True)
def slave_layer(fre2, k, theta1,theta2, c1, c2, K3):
    d_theta2 = fre2 + k / N * c1 * np.cos(theta2) - k / N * c2 * np.sin(theta2) + K3 * np.sin(theta1 - theta2)
    return d_theta2

@nb.jit(nopython=True)
def control_strength(K3,epsilon,theta1,theta2):
    d_K3=epsilon*(alpha*np.cos(theta2-theta1)-K3)
    return d_K3


@nb.jit(nopython=True)
def runge_kutta(fre1,fre2,theta1,theta2,K3):
    t1=6000
    t2 = 20000 #步长，总时长T=dt*t
    dt = 0.01
    R1 = []
    R2 = []
    for k in K:
        print(k)
        R1_transient=[]
        R2_transient=[]
        for i in range(t1):
            d1 = np.sum(np.cos(theta1))
            d2 = np.sum(np.sin(theta1))
            s1 = d1 * np.sum(np.cos(theta1))
            s2 = d2 * np.sum(np.cos(theta1))
            s3 = d1 * np.sum(np.sin(theta1))
            s4 = d2 * np.sum(np.sin(theta1))

            q1 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1)
            q2 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q1 * 0.5 * dt)
            q3 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q2 * 0.5 * dt)
            q4 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q3 * dt)
            theta1 = theta1 + (dt / 6) * (q1 + 2 * q2 + 2 * q3 + q4)
            theta1 = np.mod(theta1, 2 * np.pi)

        for i in range(t2):
            d1 = np.sum(np.cos(theta1))
            d2 = np.sum(np.sin(theta1))
            s1 = d1 * np.sum(np.cos(theta1))
            s2 = d2 * np.sum(np.cos(theta1))
            s3 = d1 * np.sum(np.sin(theta1))
            s4 = d2 * np.sum(np.sin(theta1))

            c1 = np.sum(np.sin(theta2))
            c2 = np.sum(np.cos(theta2))

            q1 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1)
            q2 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q1 * 0.5 * dt)
            q3 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q2 * 0.5 * dt)
            q4 = master_layer(fre1,d1,d2,s1,s2,s3,s4,k,theta1 + q3 * dt)

            m1 = control_strength(K3, epsilon,theta1,theta2)
            m2 = control_strength(K3+m1 * 0.5 * dt, epsilon, theta1,theta2)
            m3 = control_strength(K3+m2 * 0.5 * dt, epsilon, theta1,theta2)
            m4 = control_strength(K3+m3 * dt, epsilon, theta1,theta2)


            p1 = slave_layer(fre2, k, theta1,theta2, c1, c2, K3)
            p2 = slave_layer(fre2, k, theta1,theta2 + p1 * 0.5 * dt, c1, c2, K3)
            p3 = slave_layer(fre2, k, theta1,theta2 + p2 * 0.5 * dt, c1, c2, K3)
            p4 = slave_layer(fre2, k, theta1,theta2 + p3 * dt, c1, c2, K3)

            K3 = K3 + (dt / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
            theta1 = theta1 + (dt / 6) * (q1 + 2 * q2 + 2 * q3 + q4)
            theta2 = theta2 + (dt / 6) * (p1 + 2 * p2 + 2 * p3 + p4)

            theta1 = np.mod(theta1,2*np.pi)
            theta2 = np.mod(theta2,2*np.pi)
            r1 = ((sum(np.sin(theta1)) / N) ** 2 + (sum(np.cos(theta1)) / N) ** 2) ** 0.5
            r2 = ((sum(np.sin(theta2)) / N) ** 2 + (sum(np.cos(theta2)) / N) ** 2) ** 0.5
            R1_transient.append(r1)
            R2_transient.append(r2)
        R1.append(sum(R1_transient[-1000:])/1000)
        R2.append(sum(R2_transient[-1000:])/1000)
    return R1,R2
if __name__ == '__main__':
    #下标1对应master层，下标2对应slave层
    N=10000
    a=1
    alpha=1
    epsilon = 1
    K = np.array([3.5 - 0.05 * i for i in range(61)])
    # K = np.array([1+0.02*i for i in range(101)])
    fre1 = np.load('高斯自然频率生成.npy')
    fre2 = -fre1
    fre1 = np.array(fre1)
    fre2 = np.array(fre2)
    theta1, theta2, K3 = ini_condition()
    R1,R2= runge_kutta(fre1,fre2,theta1, theta2,K3)
    plt.figure()
    plt.plot(K, R1, '-*',color='blue')
    plt.plot(K, R2, '-*',color='red')
    np.save('person=1自然频率非爆炸a=1后向1层细致序参量.npy', R1)
    np.save('person=1自然频率非爆炸a=1后向2层细致序参量.npy', R2)
    plt.show()