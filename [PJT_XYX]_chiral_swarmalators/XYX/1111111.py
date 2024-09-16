import numpy as np

Z = np.arange(0,50,1)
L = [i*2 for i in range(10)]
X = []
for i in range(5):
    X.append(Z[L])

print(X)