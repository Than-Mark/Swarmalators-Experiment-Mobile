import os
import numpy as np
from itertools import product
from main import PeriodicalPotential
from multiprocessing import Pool

rangeLambdas = np.concatenate([
    np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
])
distanceDs = np.concatenate([
    np.arange(0.1, 1, 0.2)
])
rangeGamma = np.concatenate([
    np.arange(1.0, 11.0, 1.0)
])
kappa = [3]
period = [0.5]

savePath = "./data"

models = [
    PeriodicalPotential(l, d, g, k, L, agentsNum=200, boundaryLength=5,
              tqdm=True, savePath=savePath, overWrite=False)
    for l, d, g, k, L in product(rangeLambdas, distanceDs, rangeGamma, kappa, period)
]

finishCount = 0
sizeThres = 806

for model in models:
    if not os.path.exists(f"data/{model}.h5"):
        continue
    sizeMB = os.path.getsize(f"data/{model}.h5") / (1024 * 1024)
    if sizeMB > sizeThres:
        finishCount += 1
    if sizeMB< sizeThres:
        print(f"{model} exists but size is {sizeMB}MB")
        os.remove(f"data/{model}.h5")

print(finishCount/len(models))