import os
import numpy as np
from itertools import product
from main import ThreeBody
from multiprocessing import Pool


rangeLambdas = np.concatenate([
    np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
])
distanceDs = np.concatenate([
    np.arange(0.1, 1, 0.2)
])

savePath = "./data"

models = [
    ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
            tqdm=True, savePath=savePath, overWrite=True)
    for l1, l2, d1, d2  in product(rangeLambdas, rangeLambdas, distanceDs, distanceDs)
]

finishCount = 0
sizeThres = 113

for model in models:
    if not os.path.exists(f"data/{model}.h5"):
        continue
    sizeMB = os.path.getsize(f"data/{model}.h5") / (1024 * 1024)
    if sizeMB > sizeThres:
        finishCount += 1
    if sizeMB< sizeThres:
        print(f"{model} exists but size is {sizeMB}MB")
        # os.remove(f"data/{model}.h5")

print(finishCount/len(models))