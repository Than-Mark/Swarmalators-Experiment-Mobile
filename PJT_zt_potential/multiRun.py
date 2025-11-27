def run_model(model):
        model.run(80000)


if __name__ == "__main__":
    import numpy as np
    from itertools import product
    from main import PeriodicalPotential
    from multiprocessing import Pool
    from tqdm import tqdm

    gammas = np.linspace(0, 1, 11)
    dampingRatios = [0.1, 1, 10]
    randomSeed = 10
    strengthLambdas = np.linspace(0.1, 1.0, 5)
    distanceDs = [1.0]
    kappas = [0.25, 0.50]

    savePath = r"D:\PythonProject\System Theory\Periodical Potential\data"

    models = [
        PeriodicalPotential(
            strengthLambda=strengthLambda,
            distanceD=distanceD,
            gamma=gamma,
            dampingRatio=dampingRatio,
            kappa=kappa,
            L=1.5,
            agentsNum=1000,
            boundaryLength=5,
            dt=0.005,
            tqdm=True,
            savePath=savePath,
            overWrite=True
        )
        for strengthLambda in strengthLambdas
        for distanceD in distanceDs
        for dampingRatio in dampingRatios
        for kappa in kappas
        for gamma in gammas
    ]

    with Pool(min(len(models), 12)) as p:
        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )
