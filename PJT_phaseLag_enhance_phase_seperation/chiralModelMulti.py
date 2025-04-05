
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import sys
import shutil

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def run_model(model):
        model.run(30000)

if __name__ == "__main__":

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
    )

    @nb.njit
    def colors_idx(phaseTheta):
        return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

    from main import *
    from multiprocessing import Pool

    omegaMins = [0.1]
    phaseLags = np.linspace(-1, 1, 41) * np.pi
    randomSeeds = [4, 5, 6, 7, 8]
    boundaryLength = 7

    strengthLambda = 0.15 * 32 * 2 * 2
    distanceD0 = 1
    deltaOmega = 1

    SAVE_PATH = "./data"  # r"E:\MS_ExperimentData\general"

    models = [
        MeanFieldChiralInducedPhaseLag(
            strengthLambda=strengthLambda, distanceD0=distanceD0, boundaryLength=boundaryLength,
            phaseLag=phaseLag, 
            distribution="uniform", initPhaseTheta=np.zeros(1000),
            omegaMin=omegaMin, deltaOmega=deltaOmega,
            agentsNum=1000, savePath=SAVE_PATH, dt=0.01,
            tqdm=True, overWrite=True, randomSeed=randomSeed, shotsnaps=10
        )
        for omegaMin in omegaMins
        for phaseLag in phaseLags
        for randomSeed in randomSeeds
    ]

    with Pool(15) as p:
        # p.map(run_model, models)

        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )