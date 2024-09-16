def run_model(model):
        model.run(10)


if __name__ == "__main__":
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

    with Pool(4) as p:
        p.map(run_model, models[:4])
