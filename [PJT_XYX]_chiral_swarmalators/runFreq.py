
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import shutil

def run_model(model):
        # enhancedLambdas = np.linspace(0.009, 0.1, 50000)
        model.run(30000)
        # model.run(np.ones(60000) * model.strengthLambda)

if __name__ == "__main__":

    randomSeed = 100

    # %matplotlib inline
    # %config InlineBackend.figure_format = "svg"

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
    )

    @nb.njit
    def colors_idx(phaseTheta):
        return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

    from main import SpatialGroups, CorrectCouplingAfter, SingleDistribution, FreqAdjustableSpatialGroups
    from multiprocessing import Pool

    randomSeeds = range(20, 101, 10)
    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.005), np.arange(0.1, 1, 0.05)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.05), np.arange(1, 2.1, 0.1)
    ])

    SAVE_PATH = r"E:\MS_ExperimentData\general"

    models = [
        FreqAdjustableSpatialGroups(strengthLambda=0.4, distanceD0=0.3, omegaMin=0.5, omegaMax=omegaMax, dt=0.02, tqdm=True, savePath=SAVE_PATH, randomSeed=10, overWrite=True) 
        for omegaMax in [2, 2.5, 3.5, 4]
    ] + [
        FreqAdjustableSpatialGroups(strengthLambda=0.4, distanceD0=0.3, omegaMin=1.5, omegaMax=omegaMax, dt=0.02, tqdm=True, savePath=SAVE_PATH, randomSeed=10, overWrite=True) 
        for omegaMax in [2, 2.5, 3.5, 4]
    ]

# run_model(models[0])

    with Pool(8) as p:
        # p.map(run_model, models)

        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )